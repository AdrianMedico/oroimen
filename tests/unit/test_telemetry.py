"""Tests para Telemetry (push InfluxDB)."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hermes.config import Settings
from hermes.telemetry import Telemetry


@pytest.fixture
def settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Settings:
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token-12345")
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "test-key-67890")
    monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini-key-1234567890")
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))
    return Settings(_env_file=None)


def test_disabled_when_no_url(settings: Settings) -> None:
    monkey = settings.model_copy(update={"influx_url": "", "influx_token": ""})
    tel = Telemetry(monkey)
    assert tel.enabled is False
    # No debe lanzar error aunque lo llamemos
    tel.record_message(result="ok")
    tel.record_llm_call(model="x", status="ok", latency_ms=10, tokens_in=1, tokens_out=1)
    tel.record_tool_call(tool="y", status="ok", latency_ms=5)
    tel.record_circuit_breaker_state(model="x", state="closed")
    tel.record_heartbeat()


def test_enabled_with_creds(settings: Settings) -> None:
    monkey = settings.model_copy(
        update={"influx_url": "http://localhost:8086", "influx_token": "abc"}
    )
    with patch("hermes.telemetry.InfluxDBClient") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_write_api = MagicMock()
        mock_client.write_api.return_value = mock_write_api

        tel = Telemetry(monkey)
        assert tel.enabled is True
        assert tel._write_api is mock_write_api
        assert tel._bucket == monkey.influx_bucket


def test_record_message_calls_write(settings: Settings) -> None:
    monkey = settings.model_copy(
        update={"influx_url": "http://localhost:8086", "influx_token": "abc"}
    )
    with patch("hermes.telemetry.InfluxDBClient") as mock_client_cls:
        mock_write_api = MagicMock()
        mock_client_cls.return_value.write_api.return_value = mock_write_api

        tel = Telemetry(monkey)
        tel.record_message(result="ok")
        mock_write_api.write.assert_called_once()
        call = mock_write_api.write.call_args
        assert call.kwargs["bucket"] == monkey.influx_bucket
        # Verificamos que el point tiene la tag 'result'
        point = call.kwargs["record"]
        assert point._tags["result"] == "ok"


def test_write_failure_does_not_raise(settings: Settings) -> None:
    monkey = settings.model_copy(
        update={"influx_url": "http://localhost:8086", "influx_token": "abc"}
    )
    with patch("hermes.telemetry.InfluxDBClient") as mock_client_cls:
        mock_write_api = MagicMock()
        mock_write_api.write.side_effect = Exception("connection refused")
        mock_client_cls.return_value.write_api.return_value = mock_write_api

        tel = Telemetry(monkey)
        # No debe lanzar la excepción al caller
        tel.record_message(result="ok")


def test_aclose_closes_client(settings: Settings) -> None:
    monkey = settings.model_copy(
        update={"influx_url": "http://localhost:8086", "influx_token": "abc"}
    )
    with patch("hermes.telemetry.InfluxDBClient") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        tel = Telemetry(monkey)
        import asyncio

        asyncio.run(tel.aclose())
        mock_client.close.assert_called_once()


# ---------------------------------------------------------------------------
# Sprint 17 (F2-1): consecutive-failure counter + escalation
# ---------------------------------------------------------------------------


def _make_telemetry_with_failing_write(s: Settings) -> Telemetry:
    """Helper: telemetry con _write_api que siempre falla."""
    with patch("hermes.telemetry.InfluxDBClient") as mock_client_cls:
        mock_write_api = MagicMock()
        mock_write_api.write.side_effect = Exception("connection refused")
        mock_client_cls.return_value.write_api.return_value = mock_write_api
        return Telemetry(
            s.model_copy(update={"influx_url": "http://localhost:8086", "influx_token": "abc"})
        )


def _make_telemetry_with_working_write(s: Settings) -> Telemetry:
    """Helper: telemetry con _write_api que siempre funciona."""
    with patch("hermes.telemetry.InfluxDBClient") as mock_client_cls:
        mock_write_api = MagicMock()
        mock_client_cls.return_value.write_api.return_value = mock_write_api
        return Telemetry(
            s.model_copy(update={"influx_url": "http://localhost:8086", "influx_token": "abc"})
        )


def test_write_failure_logs_warning_per_call(
    settings: Settings, caplog: pytest.LogCaptureFixture
) -> None:
    """F2-1: cada write failed logea WARNING con conteo acumulado."""
    tel = _make_telemetry_with_failing_write(settings)
    with caplog.at_level(logging.WARNING, logger="hermes.telemetry"):
        tel.record_message(result="x")
        tel.record_message(result="x")
        tel.record_message(result="x")
    # 3 WARNINGs con consecutive_failures incrementando
    warnings = [r for r in caplog.records if "telemetry_write_failed" in r.message]
    assert len(warnings) == 3
    assert warnings[0].consecutive_failures == 1
    assert warnings[1].consecutive_failures == 2
    assert warnings[2].consecutive_failures == 3


def test_consecutive_failures_threshold_escalates_to_error(
    settings: Settings, caplog: pytest.LogCaptureFixture
) -> None:
    """F2-1: tras 5 fallos consecutivos, logea ERROR con telemetry_escalated.

    El threshold (5) y la ventana (600s) son class-level constants
    accesibles para tests via _FAILURE_ESCALATION_THRESHOLD.
    """
    tel = _make_telemetry_with_failing_write(settings)
    threshold = Telemetry._FAILURE_ESCALATION_THRESHOLD

    with caplog.at_level(logging.WARNING, logger="hermes.telemetry"):
        for _ in range(threshold):
            tel.record_message(result="x")

    # 1 ERROR con telemetry_escalated
    errors = [r for r in caplog.records if r.levelname == "ERROR"]
    assert len(errors) >= 1, f"esperaba al menos 1 ERROR, got {len(errors)}"
    escalated = [r for r in errors if "telemetry_escalated" in r.message]
    assert len(escalated) == 1
    # El ERROR incluye contexto util para el operador
    assert escalated[0].action_required
    assert "INFLUX_TOKEN" in escalated[0].action_required
    # El estado interno: escalated=True
    assert tel._escalated is True


def test_escalation_does_not_repeat_on_subsequent_failures(
    settings: Settings, caplog: pytest.LogCaptureFixture
) -> None:
    """F2-1: tras escalar, los fallos subsiguientes NO re-escalan."""
    tel = _make_telemetry_with_failing_write(settings)
    threshold = Telemetry._FAILURE_ESCALATION_THRESHOLD

    with caplog.at_level(logging.WARNING, logger="hermes.telemetry"):
        for _ in range(threshold + 10):
            tel.record_message(result="x")

    escalated = [
        r for r in caplog.records if r.levelname == "ERROR" and "telemetry_escalated" in r.message
    ]
    assert len(escalated) == 1, f"esperaba 1 sola escalación, got {len(escalated)}"


def test_recovery_after_escalation_resets_state(
    settings: Settings, caplog: pytest.LogCaptureFixture
) -> None:
    """F2-1: tras un write exitoso post-escalación, el counter se resetea
    y se logea telemetry_recovered.
    """
    s = settings.model_copy(update={"influx_url": "http://localhost:8086", "influx_token": "abc"})
    mock_write_api = MagicMock()
    mock_write_api.write.side_effect = [
        Exception("fail1"),
        Exception("fail2"),
        Exception("fail3"),
        Exception("fail4"),
        Exception("fail5"),  # 5to: triggers escalation
        None,  # 6to: success -> recovery
    ]
    with patch("hermes.telemetry.InfluxDBClient") as mock_client_cls:
        mock_client_cls.return_value.write_api.return_value = mock_write_api
        tel = Telemetry(s)
        with caplog.at_level(logging.INFO, logger="hermes.telemetry"):
            for _ in range(6):
                tel.record_message(result="x")

    assert tel._consecutive_failures == 0
    assert tel._escalated is False
    recovered = [r for r in caplog.records if "telemetry_recovered" in r.message]
    assert len(recovered) == 1


def test_window_expiry_resets_failure_tracking(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F2-1: si la ventana de tiempo expira sin suficientes failures,
    el counter se resetea y empieza fresh.
    """
    tel = _make_telemetry_with_failing_write(settings)

    # 3 fallos
    for _ in range(3):
        tel.record_message(result="x")
    assert tel._consecutive_failures == 3
    # Avanzamos el tiempo más allá de la ventana
    future = time.time() + Telemetry._FAILURE_ESCALATION_WINDOW_S + 100
    monkeypatch.setattr("hermes.telemetry.time.time", lambda: future)
    # 1 fallo mas -> deberia resetear el counter, no acumular
    tel.record_message(result="x")
    assert tel._consecutive_failures == 1
    assert tel._first_failure_at == future


def test_recovery_resets_escalated_to_allow_future_escalation(
    settings: Settings, caplog: pytest.LogCaptureFixture
) -> None:
    """F2-1: tras recovery, un nuevo outage puede volver a escalar."""
    s = settings.model_copy(update={"influx_url": "http://localhost:8086", "influx_token": "abc"})
    call_count = {"n": 0}
    mock_write_api = MagicMock()

    def write_side_effect(*args: object, **kwargs: object) -> object:
        call_count["n"] += 1
        if call_count["n"] <= 5:
            raise Exception(f"fail{call_count['n']}")
        if call_count["n"] == 6:
            return None
        raise Exception(f"second_outage{call_count['n']}")

    mock_write_api.write.side_effect = write_side_effect
    with patch("hermes.telemetry.InfluxDBClient") as mock_client_cls:
        mock_client_cls.return_value.write_api.return_value = mock_write_api
        tel = Telemetry(s)
        with caplog.at_level(logging.WARNING, logger="hermes.telemetry"):
            for _ in range(11):
                tel.record_message(result="x")

    escalated = [
        r for r in caplog.records if r.levelname == "ERROR" and "telemetry_escalated" in r.message
    ]
    assert len(escalated) == 2, f"esperaba 2 escalaciones, got {len(escalated)}"
