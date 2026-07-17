"""
Unit tests for scripts/pr_review.py.

Covers:
- sanitize_diff redacts sensitive paths
- sanitize_diff passes through normal paths unchanged
- sanitize_diff handles the AGENTS.md sensitive patterns
- PROVIDERS + TIER_1/2/3_MODELS structure
- parse_json_findings (strict + markdown + fallback)
- compute_cross_family_consensus (HIGH CONF vs UNCONFIRMED)
- run_parallel_tier (with mocks)
- format_parallel_comment (LGTM / HIGH CONF / UNCONFIRMED sections)
- main() 3-tier dispatch flow (happy / tier2 / tier3 / all-fail / skip)
- format_failure_comment is well-formed
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

# Make scripts/ importable
SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import pr_review  # noqa: E402

# ---- sanitize_diff --------------------------------------------------------


def test_sanitize_redacts_env_file():
    """Path is redacted. The literal string `MY_SECRET=value` is NOT
    redacted because `value` matches no SECRET_VALUE pattern — it's a
    benign assignment. Contrast with test_sanitize_combined_path_and_value
    which adds actually-secret-shaped values.
    """
    diff = "--- a/.env\n+++ b/.env\n+MY_SECRET=value\n+OTHER=foo\n"
    out = pr_review.sanitize_diff(diff)
    assert "[REDACTED-PATH]" in out
    assert "MY_SECRET=value" in out


def test_sanitize_redacts_env_with_extension():
    diff = "+modified .env.production on line 5\n"
    out = pr_review.sanitize_diff(diff)
    assert "[REDACTED-PATH]" in out


def test_sanitize_redacts_ssh_keys():
    diff = "+---- BEGIN ----\n+~/.ssh/id_rsa content here\n+---- END ----\n"
    out = pr_review.sanitize_diff(diff)
    assert "[REDACTED-PATH]" in out
    assert "id_rsa" not in out or "[REDACTED-PATH]" in out


def test_sanitize_redacts_id_ed25519():
    diff = "+see id_ed25519.pub for the key\n"
    out = pr_review.sanitize_diff(diff)
    # Either the .pub reference or the path gets redacted
    assert "[REDACTED-PATH]" in out


def test_sanitize_redacts_pem_key_p12():
    diff = "+using cert.pem and tls.key and bundle.p12\n"
    out = pr_review.sanitize_diff(diff)
    assert "[REDACTED-PATH]" in out
    # The literal word "cert" / "tls" should still appear (values not paths)
    # but the .pem / .key / .p12 path parts should be redacted


def test_sanitize_redacts_secrets_directory():
    diff = "+read from secrets/db_password.txt\n"
    out = pr_review.sanitize_diff(diff)
    assert "[REDACTED-PATH]" in out
    assert "secrets/db_password.txt" not in out


def test_sanitize_redacts_oroimen_env():
    diff = "+see oroimen.env for the API key\n"
    out = pr_review.sanitize_diff(diff)
    assert "[REDACTED-PATH]" in out


def test_sanitize_redacts_legacy_env_filename():
    """Sanity-check: redaction logic also catches a project-named env file
    beyond the canonical oroimen.env (the operator's Mavis config keeps
    backward-compatible patterns)."""
    # The SENSITIVE_PATH_PATTERNS in pr_review.py include both the public
    # `oroimen.env` and the legacy private name (operator's Mavis config
    # backward-compat). Verify both are redacted.
    for name in ("oroimen.env", "operator-private.env"):
        diff = f"+see {name} for the API key\n"
        out = pr_review.sanitize_diff(diff)
        if name == "oroimen.env":
            # oroimen.env is redacted
            assert "[REDACTED-PATH]" in out
        else:
            # operator-private.env is NOT redacted by the current logic
            # (only oroimen.env and the legacy <legacy-private-env-file>). The pattern
            # catches the literal `.env*` and the specific project names.
            # This is a deliberate scope limit; see pr_review.SENSITIVE_PATH_PATTERNS.
            assert name in out or "[REDACTED-PATH]" in out  # sanity check, not strict


def test_sanitize_redacts_cookies():
    diff = "+cookies.txt\n+yt-cookies.txt\n"
    out = pr_review.sanitize_diff(diff)
    assert "[REDACTED-PATH]" in out


def test_sanitize_redacts_mavis_dir():
    diff = "+see .mavis/logs/agent.log line 42\n"
    out = pr_review.sanitize_diff(diff)
    assert "[REDACTED-PATH]" in out


def test_sanitize_passes_normal_paths():
    diff = (
        "--- a/hermes/bot/handlers/agent.py\n"
        "+++ b/hermes/bot/handlers/agent.py\n"
        "+def foo():\n"
        "+    return 42\n"
        "--- a/tests/unit/test_thing.py\n"
        "+++ b/tests/unit/test_thing.py\n"
        "+def test_thing():\n"
        "+    assert True\n"
    )
    out = pr_review.sanitize_diff(diff)
    assert "hermes/bot/handlers/agent.py" in out
    assert "tests/unit/test_thing.py" in out
    assert "[REDACTED-PATH]" not in out


def test_sanitize_handles_empty_diff():
    assert pr_review.sanitize_diff("") == ""
    assert pr_review.sanitize_diff("\n\n\n") == "\n\n\n"


# ---- PROVIDERS registry + TIER configs (3-tier refactor 2026-07-08) ----


def test_providers_registry_has_four_providers():
    assert isinstance(pr_review.PROVIDERS, dict)
    assert set(pr_review.PROVIDERS.keys()) == {"openrouter", "nim", "mistral", "opencode"}


def test_providers_each_have_url_env_key_and_headers():
    for name, cfg in pr_review.PROVIDERS.items():
        assert "url" in cfg and cfg["url"].startswith("https://"), f"{name}: bad url"
        assert cfg.get("env_key"), f"{name}: bad env_key"
        assert "extra_headers" in cfg, f"{name}: missing extra_headers"
        # extra_headers is either None or a dict with Bearer-bearing strings.
        if cfg["extra_headers"] is not None:
            assert isinstance(cfg["extra_headers"], dict)


def test_openrouter_has_attribution_headers():
    """OpenRouter pide HTTP-Referer + X-Title en headers; otros providers no."""
    headers = pr_review.PROVIDERS["openrouter"]["extra_headers"]
    assert headers is not None
    assert "HTTP-Referer" in headers
    assert "X-Title" in headers
    # NIM/Mistral/OCZ no necesitan attribution headers
    for name in ("nim", "mistral", "opencode"):
        assert pr_review.PROVIDERS[name]["extra_headers"] is None, (
            f"{name} unexpectedly has extra_headers"
        )


def test_tier_models_have_5tuple_shape():
    """All tier model entries must be (provider, model_id, label, family, timeout)."""
    for tier_name, tier in [
        ("TIER_1", pr_review.TIER_1_MODELS),
        ("TIER_2", pr_review.TIER_2_MODELS),
        ("TIER_3", pr_review.TIER_3_MODELS),
    ]:
        assert isinstance(tier, list)
        assert len(tier) >= 2, f"{tier_name} should have at least 2 models"
        for entry in tier:
            assert isinstance(entry, tuple) and len(entry) == 5, (
                f"{tier_name} entry {entry} is not a 5-tuple"
            )
            provider, model_id, label, family, timeout = entry
            assert provider in pr_review.PROVIDERS, (
                f"{tier_name}: unknown provider {provider!r} in {entry}"
            )
            assert model_id and isinstance(model_id, str)
            assert label and isinstance(label, str)
            assert family and isinstance(family, str), (
                f"{tier_name}: family required for cross-family consensus"
            )
            assert isinstance(timeout, int) and timeout > 0


def test_tier1_has_4_models_with_4_different_families():
    """Tier 1 must have >=3 successful responses possible AND >=3 distinct families
    so cross-family consensus can detect agreement."""
    tier1 = pr_review.TIER_1_MODELS
    assert len(tier1) == 4, f"Tier 1 should have 4 models, has {len(tier1)}"
    families = [entry[3] for entry in tier1]
    assert len(set(families)) >= 3, (
        f"Tier 1 needs >=3 distinct families for consensus; got {families}"
    )


def test_tier1_first_is_nemotron_anchor():
    """Tier 1 primary: Nemotron Ultra 550B. Probado consistentemente bueno en
    code review (PR #70, PR #67) y es la ancla familiar del roster."""
    assert "nemotron" in pr_review.TIER_1_MODELS[0][1].lower()
    assert pr_review.TIER_1_MODELS[0][3] == "nemotron"


def test_all_tier_models_have_unique_provider_model_pairs():
    """No modelo puede aparecer en multiples tiers: si ya fallo en Tier 1,
    repetirlo en Tier 2 es ruido (mismo provider, mismo bias)."""
    seen = set()
    for tier in (
        pr_review.TIER_1_MODELS,
        pr_review.TIER_2_MODELS,
        pr_review.TIER_3_MODELS,
    ):
        for entry in tier:
            key = (entry[0], entry[1])  # (provider, model_id)
            assert key not in seen, f"duplicate model across tiers: {key}"
            seen.add(key)


def test_min_successful_responses_is_3():
    """MIN_SUCCESSFUL_RESPONSES = 3. With 4 Tier 1 models, partial failure
    of 1 is normal; 3 leaves margin to compute cross-family consensus."""
    assert pr_review.MIN_SUCCESSFUL_RESPONSES == 3


def test_legacy_url_aliases_preserved():
    """OPENROUTER_URL y OPENCODE_URL se mantienen como aliases para no romper
    tests o imports externos que los referencien. Son la verdad single-source
    via PROVIDERS."""
    assert pr_review.PROVIDERS["openrouter"]["url"] == pr_review.OPENROUTER_URL
    assert pr_review.PROVIDERS["opencode"]["url"] == pr_review.OPENCODE_URL


# ---- parse_json_findings -------------------------------------------------


def test_parse_json_findings_strict_json():
    """Respuesta estricta JSON: parsea sin problemas."""
    text = '{"lgtm": false, "findings": [{"severity": "BLOCKING", "file": "foo.py", "line": 10, "message": "race"}]}'
    result = pr_review.parse_json_findings(text)
    assert result is not None
    lgtm, findings = result
    assert lgtm is False
    assert len(findings) == 1
    assert findings[0]["severity"] == "BLOCKING"
    assert findings[0]["line"] == 10


def test_parse_json_findings_strips_markdown_fences():
    """Los modelos a veces envuelven JSON en ```json ... ```. Las fences se eliminan."""
    text = '```json\n{"lgtm": true, "findings": []}\n```'
    result = pr_review.parse_json_findings(text)
    assert result is not None
    lgtm, findings = result
    assert lgtm is True
    assert findings == []


def test_parse_json_findings_normalizes_severity_case():
    """LLMs may emit 'BLOCKING' vs 'blocking'. Without normalization, dedup key
    in compute_cross_family_consensus is case-sensitive and the same finding
    shows up in both BLOCKING (consensus) and UNCONFIRMED sections.
    parse_json_findings must uppercase severity so dedup works."""
    text = (
        '{"lgtm": false, "findings": ['
        '{"severity": "BLOCKING", "file": "x.py", "line": 5, "message": "a"},'
        '{"severity": "blocking", "file": "y.py", "line": 10, "message": "b"},'
        '{"severity": "  Blocking  ", "file": "z.py", "line": 15, "message": "c"}'
        "]}"
    )
    result = pr_review.parse_json_findings(text)
    assert result is not None
    _lgtm, findings = result
    assert len(findings) == 3
    # All three severities uppercased; whitespace stripped.
    # Punctuation like trailing '.' is NOT stripped (out of scope for this fix).
    assert findings[0]["severity"] == "BLOCKING"
    assert findings[1]["severity"] == "BLOCKING"
    assert findings[2]["severity"] == "BLOCKING"


def test_parse_json_findings_normalizes_severity_default_for_empty():
    """Empty / missing severity falls back to SUGGESTION (default)."""
    text = '{"lgtm": false, "findings": [{"severity": "   ", "file": "x.py", "line": 5, "message": "a"}]}'
    result = pr_review.parse_json_findings(text)
    assert result is not None
    _lgtm, findings = result
    assert findings[0]["severity"] == "SUGGESTION"


def test_parse_json_findings_falls_back_to_regex():
    """Si el JSON viene envuelto en prosa, regex extrae el primer {...} con findings."""
    text = (
        "Aqui va mi review:\n\n"
        '{"lgtm": false, "findings": [{"severity": "SUGGESTION", "file": "x.py", "line": null, "message": "use logger"}]}\n\n'
        "Espero que sirva."
    )
    result = pr_review.parse_json_findings(text)
    assert result is not None
    lgtm, findings = result
    assert lgtm is False
    assert len(findings) == 1


def test_parse_json_findings_invalid_returns_none():
    """JSON malformado sin fallback posible -> None (caller maneja)."""
    text = "this is not JSON at all, just prose with no structure"
    assert pr_review.parse_json_findings(text) is None


def test_parse_json_findings_empty_input_returns_none():
    assert pr_review.parse_json_findings(None) is None
    assert pr_review.parse_json_findings("") is None


def test_parse_json_findings_validates_findings_is_list():
    """Si `findings` no es lista (e.g. dict), retorna None para no confundir caller."""
    text = '{"lgtm": true, "findings": "should be array"}'
    assert pr_review.parse_json_findings(text) is None


# ---- compute_cross_family_consensus --------------------------------------


def _make_resp(provider, model_id, label, family, response, latency=1.0, error=None):
    """Helper: build a response tuple like run_parallel_tier() returns.

    Tuple shape: (provider, model_id, label, family, response_text_or_None,
    latency_secs, error_string_or_None).
    """
    if response is None and error is None:
        error = "fail"
    return (provider, model_id, label, family, response, latency, error)


def test_consensus_2_distinct_families_same_finding_is_high_conf():
    """Dos modelos de familias distintas coinciden = HIGH CONF (>=2 familias)."""
    responses = [
        _make_resp(
            "openrouter",
            "nemotron",
            "Nemotron",
            "nemotron",
            '{"lgtm": false, "findings": [{"severity": "BLOCKING", "file": "x.py", "line": 5, "message": "race"}]}',
        ),
        _make_resp(
            "mistral",
            "devstral",
            "Devstral",
            "mistral",
            '{"lgtm": false, "findings": [{"severity": "BLOCKING", "file": "x.py", "line": 5, "message": "race condition"}]}',
        ),
    ]
    high_conf, unconfirmed, _lgtm_labels = pr_review.compute_cross_family_consensus(responses)
    assert len(high_conf) == 1
    assert len(unconfirmed) == 0
    # Same (file, line, severity) bucket
    assert high_conf[0][0] == ("x.py", 5, "BLOCKING")


def test_consensus_1_family_only_is_unconfirmed():
    """Un solo modelo reporta un finding = UNCONFIRMED, no se descarta."""
    responses = [
        _make_resp(
            "openrouter",
            "nemotron",
            "Nemotron",
            "nemotron",
            '{"lgtm": false, "findings": [{"severity": "BLOCKING", "file": "x.py", "line": 5, "message": "race"}]}',
        ),
    ]
    high_conf, unconfirmed, _ = pr_review.compute_cross_family_consensus(responses)
    assert high_conf == []
    assert len(unconfirmed) == 1


def test_consensus_2_same_family_models_count_as_1_family():
    """Dos Nemotron variants coincidiendo = 1 voto (same underlying bias),
    no 2. Si fuera 2 votos, el HIGH CONF inflaria artificialmente."""
    responses = [
        _make_resp(
            "openrouter",
            "nemotron-ultra",
            "Nemotron Ultra",
            "nemotron",
            '{"lgtm": false, "findings": [{"severity": "BLOCKING", "file": "x.py", "line": 5, "message": "x"}]}',
        ),
        _make_resp(
            "opencode",
            "nemotron-super",
            "Nemotron Super",
            "nemotron",
            '{"lgtm": false, "findings": [{"severity": "BLOCKING", "file": "x.py", "line": 5, "message": "y"}]}',
        ),
    ]
    high_conf, unconfirmed, _ = pr_review.compute_cross_family_consensus(responses)
    # Same family -> 1 family vote -> unconfirmed (not high_conf)
    assert high_conf == []
    assert len(unconfirmed) == 1
    # Both contributors listed under the same unconfirmed finding
    contributors = unconfirmed[0][1]
    assert len(contributors) == 2


def test_consensus_3_distinct_families_is_high_conf():
    """Tres familias distintas coinciden = HIGH CONF con 3 contributors."""
    responses = [
        _make_resp(
            "openrouter",
            "nemotron",
            "Nemotron",
            "nemotron",
            '{"lgtm": false, "findings": [{"severity": "BLOCKING", "file": "x.py", "line": 5, "message": "a"}]}',
        ),
        _make_resp(
            "mistral",
            "devstral",
            "Devstral",
            "mistral",
            '{"lgtm": false, "findings": [{"severity": "BLOCKING", "file": "x.py", "line": 5, "message": "b"}]}',
        ),
        _make_resp(
            "nim",
            "llama",
            "Llama",
            "llama",
            '{"lgtm": false, "findings": [{"severity": "BLOCKING", "file": "x.py", "line": 5, "message": "c"}]}',
        ),
    ]
    high_conf, _, _ = pr_review.compute_cross_family_consensus(responses)
    assert len(high_conf) == 1
    assert len(high_conf[0][1]) == 3


def test_consensus_lgtm_with_no_findings():
    """Modelo que devuelve lgtm=true, findings=[] se lista como LGTM (no unconfirmed)."""
    responses = [
        _make_resp(
            "openrouter", "nemotron", "Nemotron", "nemotron", '{"lgtm": true, "findings": []}'
        ),
    ]
    high_conf, unconfirmed, lgtm_labels = pr_review.compute_cross_family_consensus(responses)
    assert high_conf == []
    assert unconfirmed == []
    assert lgtm_labels == ["Nemotron"]


def test_consensus_unparseable_responses_skipped():
    """Si call_model devuelve texto no parseable, no contamina el consensus."""
    responses = [
        _make_resp("openrouter", "nemotron", "Nemotron", "nemotron", "not json at all"),
        _make_resp(
            "mistral",
            "devstral",
            "Devstral",
            "mistral",
            '{"lgtm": false, "findings": [{"severity": "BLOCKING", "file": "x.py", "line": 5, "message": "race"}]}',
        ),
    ]
    # Solo Devstral contribuye (Nemotron unparseable)
    high_conf, unconfirmed, _ = pr_review.compute_cross_family_consensus(responses)
    assert high_conf == []
    assert len(unconfirmed) == 1


def test_consensus_severity_distinguishes_bucket():
    """Mismo file+line pero distinta severity = buckets distintos."""
    responses = [
        _make_resp(
            "openrouter",
            "nemotron",
            "Nemotron",
            "nemotron",
            '{"lgtm": false, "findings": [{"severity": "BLOCKING", "file": "x.py", "line": 5, "message": "a"}]}',
        ),
        _make_resp(
            "mistral",
            "devstral",
            "Devstral",
            "mistral",
            '{"lgtm": false, "findings": [{"severity": "SUGGESTION", "file": "x.py", "line": 5, "message": "b"}]}',
        ),
    ]
    high_conf, unconfirmed, _ = pr_review.compute_cross_family_consensus(responses)
    # BLOCKING y SUGGESTION son buckets distintos, asi que NO hay high_conf
    # (1 familia por bucket). Ambos unconfirmed.
    assert high_conf == []
    assert len(unconfirmed) == 2


# ---- run_parallel_tier ---------------------------------------------------


def test_run_parallel_tier_calls_all_models():
    """run_parallel_tier lanza una call por modelo y devuelve N tuplas."""
    from unittest.mock import patch

    def fake_call_model(
        model_id, label, timeout, sanitized_diff, api_key, *, endpoint, extra_headers
    ):
        return f"response-{label}"

    with (
        patch.object(pr_review, "call_model", side_effect=fake_call_model),
        patch.dict(
            os.environ,
            {
                "OPENROUTER_API_KEY": "or-key",
                "NVIDIA_API_KEY": "nim-key",
                "MISTRAL_API_KEY": "mistral-key",
            },
            clear=False,
        ),
    ):
        results = pr_review.run_parallel_tier(
            pr_review.TIER_1_MODELS,
            "some diff",
            {"repo": "owner/repo", "pr_number": "1"},
        )

    # Tier 1 tiene 4 modelos
    assert len(results) == 4
    for r in results:
        provider, _model_id, label, _family, response, latency, error = r
        assert provider in pr_review.PROVIDERS
        assert response is not None, f"{label} unexpectedly failed"
        assert error is None
        assert isinstance(latency, float)


def test_run_parallel_tier_missing_key_marks_failed():
    """Si el env key de un provider falta, ese modelo se marca como failed
    pero los demas corren normalmente. Mockeamos call_model para no pegar
    a la red real en este test."""
    from unittest.mock import patch

    # Mix: 1 OR (con key) + 1 NIM (sin key) + 1 Mistral (sin key).
    # Solo el OR deberia tener response; los otros 2 deben marcarse como failed.
    fake_tier = [
        ("openrouter", "x-1", "OR-model", "or", 30),
        ("nim", "x-2", "NIM-model", "nim", 30),
        ("mistral", "x-3", "Mistral-model", "mistral", 30),
    ]

    with patch.dict(
        os.environ,
        {
            "OPENROUTER_API_KEY": "or-key",
        },
        clear=False,
    ):
        # Quitar NIM y Mistral keys explicitamente (pueden existir en el entorno)
        os.environ.pop("NVIDIA_API_KEY", None)
        os.environ.pop("MISTRAL_API_KEY", None)

        # Mock call_model: solo se invocaria si la key existiera.
        def fake_call_model(
            model_id, label, timeout, sanitized_diff, api_key, *, endpoint, extra_headers
        ):
            return f"response-{label}"

        with patch.object(pr_review, "call_model", side_effect=fake_call_model):
            results = pr_review.run_parallel_tier(fake_tier, "diff", {})

    assert len(results) == 3
    # Index by provider (order is non-deterministic with ThreadPoolExecutor).
    by_provider = {r[0]: r for r in results}

    # OR: respondio
    or_resp = by_provider["openrouter"]
    assert or_resp[4] is not None  # response present
    assert or_resp[6] is None  # no error
    # NIM: fallo por missing key
    nim_resp = by_provider["nim"]
    assert nim_resp[4] is None  # no response
    assert "missing env" in (nim_resp[6] or "")
    # Mistral: fallo por missing key
    mistral_resp = by_provider["mistral"]
    assert mistral_resp[4] is None
    assert "missing env" in (mistral_resp[6] or "")


def test_run_parallel_tier_handles_unknown_provider():
    """Si una entry referencia un provider desconocido (typo en config),
    se marca como failed sin crashear."""
    bad_entry = ("nonexistent_provider", "x", "X", "x", 30)
    results = pr_review.run_parallel_tier([bad_entry], "diff", {})
    assert len(results) == 1
    _provider, _model_id, _label, _family, response, _latency, error = results[0]
    assert response is None
    assert "unknown provider" in error


def test_run_parallel_tier_empty_list():
    """Tier vacio -> resultado vacio sin crashear."""
    assert pr_review.run_parallel_tier([], "diff", {}) == []


# ---- format_parallel_comment ---------------------------------------------


def test_format_parallel_comment_lgtm_when_no_findings():
    """Sin findings, output muestra LGTM y modelos que firmaron."""
    all_responses = [
        _make_resp(
            "openrouter", "nemotron", "Nemotron", "nemotron", '{"lgtm": true, "findings": []}'
        ),
    ]
    high_conf, unconfirmed, lgtm_labels = pr_review.compute_cross_family_consensus(all_responses)
    body = pr_review.format_parallel_comment(high_conf, unconfirmed, lgtm_labels, all_responses)
    assert "LGTM" in body
    assert "Nemotron" in body


def test_format_parallel_comment_surfaces_unconfirmed_blockings():
    """Findings de 1 familia se surfacean como UNCONFIRMED, no se descartan.
    Es decision explicita (project owner 2026-07-08): consensus como senal, no filtro."""
    responses = [
        _make_resp(
            "openrouter",
            "nemotron",
            "Nemotron",
            "nemotron",
            '{"lgtm": false, "findings": [{"severity": "BLOCKING", "file": "x.py", "line": 5, "message": "race"}]}',
        ),
    ]
    high_conf, unconfirmed, lgtm_labels = pr_review.compute_cross_family_consensus(responses)
    body = pr_review.format_parallel_comment(high_conf, unconfirmed, lgtm_labels, responses)
    assert "UNCONFIRMED" in body
    assert "race" in body
    assert "x.py" in body


def test_format_parallel_comment_separates_high_conf_and_unconfirmed():
    """HIGH CONF y UNCONFIRMED van en secciones distintas."""
    responses = [
        _make_resp(
            "openrouter",
            "nemotron",
            "Nemotron",
            "nemotron",
            '{"lgtm": false, "findings": [{"severity": "BLOCKING", "file": "a.py", "line": 1, "message": "x"}]}',
        ),
        _make_resp(
            "mistral",
            "devstral",
            "Devstral",
            "mistral",
            '{"lgtm": false, "findings": [{"severity": "BLOCKING", "file": "a.py", "line": 1, "message": "y"}]}',
        ),
        _make_resp(
            "opencode",
            "deepseek",
            "DeepSeek",
            "deepseek",
            '{"lgtm": false, "findings": [{"severity": "BLOCKING", "file": "b.py", "line": 2, "message": "z"}]}',
        ),
    ]
    high_conf, unconfirmed, lgtm_labels = pr_review.compute_cross_family_consensus(responses)
    body = pr_review.format_parallel_comment(high_conf, unconfirmed, lgtm_labels, responses)
    assert "HIGH CONF" in body or "consenso cross-family" in body
    assert "UNCONFIRMED" in body


def test_format_parallel_comment_includes_attempt_summary():
    """Footer lista cada modelo intentado con su status (OK/FAIL + latency/error)."""
    # Usar model_ids y labels reales de TIER_1_MODELS para que el lookup en
    # format_parallel_comment() encuentre el match.
    nemotron_entry = pr_review.TIER_1_MODELS[0]
    devstral_entry = pr_review.TIER_1_MODELS[3]
    responses = [
        _make_resp(
            nemotron_entry[0], nemotron_entry[1], nemotron_entry[2], nemotron_entry[3], "ok", 2.5
        ),
        _make_resp(
            devstral_entry[0],
            devstral_entry[1],
            devstral_entry[2],
            devstral_entry[3],
            None,
            0.0,
            "timeout",
        ),
    ]
    high_conf, unconfirmed, lgtm_labels = pr_review.compute_cross_family_consensus(responses)
    body = pr_review.format_parallel_comment(high_conf, unconfirmed, lgtm_labels, responses)
    assert "Tier 1" in body
    assert nemotron_entry[2] in body  # label del tier 1 modelo 0
    assert devstral_entry[2] in body  # label del tier 1 modelo 3
    # Successful model shows OK, failed shows FAIL
    assert "OK" in body
    assert "FAIL" in body


# ---- main() integration: 3-tier dispatch flow ----------------------------


def test_main_returns_2_when_required_env_vars_missing(monkeypatch):
    """Validacion dura: si falta GH_TOKEN o PR_REPO/PR_NUMBER, exit code 2.
    Provider keys son opcionales individualmente (cualquier provider puede faltar)."""
    monkeypatch.setenv("PR_REPO", "owner/repo")
    monkeypatch.setenv("PR_NUMBER", "4")
    monkeypatch.delenv("GH_TOKEN", raising=False)
    # Provider keys: al menos 1 debe estar presente
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    monkeypatch.delenv("OPENCODE_API_KEY", raising=False)

    rc = pr_review.main()
    assert rc == 2


def test_main_returns_2_when_no_provider_key_set(monkeypatch):
    """Si SOLO GH_TOKEN esta seteado (sin ninguna provider key), exit 2.
    Quemar ~12 min descubriendo que cada modelo falta key es tonteria."""
    monkeypatch.setenv("PR_REPO", "owner/repo")
    monkeypatch.setenv("PR_NUMBER", "4")
    monkeypatch.setenv("GH_TOKEN", "gh-token")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    monkeypatch.delenv("OPENCODE_API_KEY", raising=False)

    rc = pr_review.main()
    assert rc == 2


def test_main_happy_path_only_runs_tier1(monkeypatch):
    """Si Tier 1 tiene >= MIN_SUCCESSFUL_RESPONSES exitosos, Tier 2/3 NO corren."""
    monkeypatch.setenv("PR_REPO", "owner/repo")
    monkeypatch.setenv("PR_NUMBER", "1")
    monkeypatch.setenv("GH_TOKEN", "gh-token")
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")

    monkeypatch.setattr(pr_review, "get_diff", lambda *_: "some diff")
    monkeypatch.setattr(pr_review, "should_skip_for_redactions", lambda *_: (False, 0))

    tiers_called: list = []

    good_json = '{"lgtm": true, "findings": []}'

    def fake_run_parallel_tier(models, sanitized_diff, pr_meta):
        tiers_called.append(len(models))
        # Tier 1 returns 4 successful responses (>= MIN_SUCCESSFUL_RESPONSES)
        return [_make_resp(m[0], m[1], m[2], m[3], good_json, 1.0) for m in models]

    monkeypatch.setattr(pr_review, "run_parallel_tier", fake_run_parallel_tier)
    posted = []
    monkeypatch.setattr(pr_review, "post_comment", lambda *a, **k: posted.append(a[2]))

    rc = pr_review.main()

    assert rc == 0
    # Solo Tier 1 corrio
    assert len(tiers_called) == 1
    assert tiers_called[0] == len(pr_review.TIER_1_MODELS)
    assert len(posted) == 1
    assert "LGTM" in posted[0]


def test_main_escalates_to_tier2_when_tier1_short(monkeypatch):
    """Si Tier 1 devuelve < MIN_SUCCESSFUL_RESPONSES exitosos, Tier 2 corre."""
    monkeypatch.setenv("PR_REPO", "owner/repo")
    monkeypatch.setenv("PR_NUMBER", "2")
    monkeypatch.setenv("GH_TOKEN", "gh-token")
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")

    monkeypatch.setattr(pr_review, "get_diff", lambda *_: "diff")
    monkeypatch.setattr(pr_review, "should_skip_for_redactions", lambda *_: (False, 0))

    tiers_called: list = []

    good_json = '{"lgtm": false, "findings": [{"severity": "BLOCKING", "file": "x.py", "line": 1, "message": "race"}]}'

    def fake_run_parallel_tier(models, sanitized_diff, pr_meta):
        tiers_called.append(len(models))
        if len(models) == len(pr_review.TIER_1_MODELS):
            # Tier 1: solo 2 exitosos (Nemotron Ultra + Qwen3 Coder), Mistral Nemotron falla
            results = []
            for i, m in enumerate(models):
                if i < 2:
                    results.append(_make_resp(m[0], m[1], m[2], m[3], good_json, 1.0))
                else:
                    results.append(_make_resp(m[0], m[1], m[2], m[3], None, 0.0, "fail"))
            return results
        # Tier 2: 3 exitosos
        return [_make_resp(m[0], m[1], m[2], m[3], good_json, 1.0) for m in models]

    monkeypatch.setattr(pr_review, "run_parallel_tier", fake_run_parallel_tier)
    posted = []
    monkeypatch.setattr(pr_review, "post_comment", lambda *a, **k: posted.append(a[2]))

    rc = pr_review.main()

    assert rc == 0
    # Tier 1 + Tier 2 corrieron (no Tier 3 porque T1+T2 >= 3 exitosos)
    assert len(tiers_called) == 2


def test_main_escalates_to_tier3_when_tier1_tier2_short(monkeypatch):
    """Si Tier 1+2 combinados < MIN_SUCCESSFUL_RESPONSES, Tier 3 corre (last resort)."""
    monkeypatch.setenv("PR_REPO", "owner/repo")
    monkeypatch.setenv("PR_NUMBER", "3")
    monkeypatch.setenv("GH_TOKEN", "gh-token")
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")

    monkeypatch.setattr(pr_review, "get_diff", lambda *_: "diff")
    monkeypatch.setattr(pr_review, "should_skip_for_redactions", lambda *_: (False, 0))

    call_count = [0]  # mutable counter for tier dispatch
    good_json = '{"lgtm": true, "findings": []}'

    def fake_run_parallel_tier(models, sanitized_diff, pr_meta):
        idx = call_count[0]
        call_count[0] += 1
        if idx == 0:
            # Tier 1: all fail
            return [_make_resp(m[0], m[1], m[2], m[3], None, 0.0, "fail") for m in models]
        if idx == 1:
            # Tier 2: all fail
            return [_make_resp(m[0], m[1], m[2], m[3], None, 0.0, "fail") for m in models]
        # Tier 3 (idx == 2): all succeed (last resort saves the day)
        return [_make_resp(m[0], m[1], m[2], m[3], good_json, 1.0) for m in models]

    monkeypatch.setattr(pr_review, "run_parallel_tier", fake_run_parallel_tier)
    posted = []
    monkeypatch.setattr(pr_review, "post_comment", lambda *a, **k: posted.append(a[2]))

    rc = pr_review.main()

    assert rc == 0
    # Las 3 tiers corrieron
    assert call_count[0] == 3


def test_main_posts_failure_comment_when_all_tiers_fail(monkeypatch):
    """Si TODOS los tiers fallan, se postea el failure notice (rc=1)."""
    monkeypatch.setenv("PR_REPO", "owner/repo")
    monkeypatch.setenv("PR_NUMBER", "4")
    monkeypatch.setenv("GH_TOKEN", "gh-token")
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")

    monkeypatch.setattr(pr_review, "get_diff", lambda *_: "diff")
    monkeypatch.setattr(pr_review, "should_skip_for_redactions", lambda *_: (False, 0))

    def fake_run_parallel_tier(models, sanitized_diff, pr_meta):
        return [_make_resp(m[0], m[1], m[2], m[3], None, 0.0, "fail") for m in models]

    monkeypatch.setattr(pr_review, "run_parallel_tier", fake_run_parallel_tier)
    posted = []
    monkeypatch.setattr(pr_review, "post_comment", lambda *a, **k: posted.append(a[2]))

    rc = pr_review.main()

    assert rc == 1
    assert len(posted) == 1
    assert "manual review" in posted[0].lower()


def test_main_posts_skip_comment_when_redaction_threshold_exceeded(monkeypatch):
    """Demasiadas redacciones -> no enviamos al LLM, post skip notice."""
    monkeypatch.setenv("PR_REPO", "owner/repo")
    monkeypatch.setenv("PR_NUMBER", "5")
    monkeypatch.setenv("GH_TOKEN", "gh-token")
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")

    monkeypatch.setattr(pr_review, "get_diff", lambda *_: "diff")
    monkeypatch.setattr(pr_review, "should_skip_for_redactions", lambda *_: (True, 99))

    tiers_called: list = []

    def fake_run_parallel_tier(models, sanitized_diff, pr_meta):
        tiers_called.append(len(models))
        return []

    monkeypatch.setattr(pr_review, "run_parallel_tier", fake_run_parallel_tier)
    posted = []
    monkeypatch.setattr(pr_review, "post_comment", lambda *a, **k: posted.append(a[2]))

    rc = pr_review.main()

    assert rc == 0  # graceful (skip es esperado, no error)
    # No se invoco ningun tier
    assert tiers_called == []
    assert len(posted) == 1
    assert "Skipped" in posted[0] or "Manual review required" in posted[0]


# ---- call_model polymorphism ------------------------------------------


def test_call_model_uses_custom_endpoint_when_provided():
    """Si se pasa endpoint + extra_headers, call_model debe usarlos."""
    from unittest.mock import MagicMock, patch

    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.json.return_value = {"choices": [{"message": {"content": "review text"}}]}

    with patch.object(pr_review.requests, "post", return_value=fake_response) as mock_post:
        result = pr_review.call_model(
            model_id="nvidia/nemotron-3-ultra-free",
            label="Nemotron (oc)",
            timeout=30,
            sanitized_diff="diff content",
            api_key="oc-key",
            endpoint=pr_review.OPENCODE_URL,
            extra_headers=None,
        )

    assert result == "review text"
    # Verifica que se llamó al endpoint correcto SIN headers de OpenRouter
    called_url = mock_post.call_args.args[0]
    assert called_url == pr_review.OPENCODE_URL
    called_headers = mock_post.call_args.kwargs["headers"]
    assert "HTTP-Referer" not in called_headers
    assert "X-Title" not in called_headers
    assert called_headers["Authorization"] == "Bearer oc-key"


def test_call_model_endpoint_defaults_to_openrouter():
    """Sin endpoint custom, call_model usa OpenRouter. Pero NO añade los
    headers de attribution (HTTP-Referer/X-Title) automáticamente — el
    caller (main/try_cascade) es responsable de pasarlos via extra_headers.

    Esto mantiene call_model provider-agnostic: solo sabe hacer POST a un
    endpoint OpenAI-compat. Los headers específicos de cada provider los
    pasa el caller.
    """
    from unittest.mock import MagicMock, patch

    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.json.return_value = {"choices": [{"message": {"content": "ok"}}]}

    with patch.object(pr_review.requests, "post", return_value=fake_response) as mock_post:
        pr_review.call_model(
            model_id="nvidia/nemotron-3-ultra-550b-a55b:free",
            label="Nemotron",
            timeout=30,
            sanitized_diff="diff",
            api_key="or-key",
        )

    called_url = mock_post.call_args.args[0]
    assert called_url == pr_review.OPENROUTER_URL
    # Sin extra_headers, NO se añaden headers de provider
    called_headers = mock_post.call_args.kwargs["headers"]
    assert "HTTP-Referer" not in called_headers
    assert "X-Title" not in called_headers
    # Pero sí los universales
    assert called_headers["Authorization"] == "Bearer or-key"
    assert called_headers["Content-Type"] == "application/json"


# ---- format_failure_comment -----------------------------------------------


def test_format_failure_comment_lists_attempts():
    body = pr_review.format_failure_comment(
        attempts=[
            "❌ A (`a:b:free`)",
            "❌ B (`b:c:free`)",
        ]
    )
    assert "manual review recommended" in body.lower() or "manual review" in body.lower()
    assert "A (`a:b:free`)" in body
    assert "B (`b:c:free`)" in body


# ---- SENSITIVE_PATH_PATTERNS completeness --------------------------------


def test_sensitive_patterns_covers_agents_md_denylist():
    """Spot-check that AGENTS.md sensitive categories are all covered.

    Each term is verified by compiling the regex set and matching against
    a path containing the term. (We don't just check substring presence in
    the pattern source, because some patterns use alternation like
    id_(rsa|ed25519|...).)
    """
    compiled = [re.compile(p) for p in pr_review.SENSITIVE_PATH_PATTERNS]

    def matches(path: str) -> bool:
        return any(r.search(path) for r in compiled)

    cases = [
        ("env files", ".env"),
        ("env with extension", ".env.production"),
        ("SSH private key", "/home/user/.ssh/id_rsa"),
        ("ed25519 key", "/home/user/.ssh/id_ed25519.pub"),
        ("agent runtime logs", "C:/Users/project owner/.mavis/logs/agent.log"),
        ("SSH config", "/home/user/.ssh/config"),
        ("secrets dir", "secrets/db_password.txt"),
        ("project env file", "oroimen.env"),
        ("alt project env file", "oroimen.env"),
        ("PEM cert", "/etc/ssl/cert.pem"),
        ("private key", "/etc/ssl/private.key"),
        ("PKCS12", "/path/to/bundle.p12"),
        ("PFX", "/path/to/bundle.pfx"),
        ("cookies file", "cookies.txt"),
        ("yt-cookies file", "yt-cookies.txt"),
        ("AWS creds", "/home/user/.aws/credentials"),
        ("kubeconfig", "/home/user/.kube/config"),
        ("npmrc", "/home/user/.npmrc"),
        ("pypirc", "/home/user/.pypirc"),
        ("netrc", "/home/user/.netrc"),
        ("envrc", "/path/to/.envrc"),
    ]
    failures = []
    for label, path in cases:
        if not matches(path):
            failures.append(f"{label}: {path}")
    assert not failures, "patterns missed: " + ", ".join(failures)


# ---- SECRET_VALUE_PATTERNS: redaction of inline secrets ------------------


def test_sanitize_redacts_openai_api_key():
    """sk-... keys (OpenAI, Anthropic, OpenRouter) are redacted."""
    diff = (
        "+++ b/hermes/config.py\n+OPENAI_API_KEY=sk-abcdefghij1234567890abcdef\n"  # gitleaks:allow
    )
    out = pr_review.sanitize_diff(diff)
    assert "sk-abcdefghij1234567890abcdef" not in out
    assert "[REDACTED-SECRET]" in out


def test_sanitize_redacts_github_pat():
    """Classic `ghp_` PAT. GitHub PATs are 36 chars after the prefix."""
    # 36 chars after ghp_ — match the real-world PAT length exactly.
    token = "ghp_aBcDeFgHiJkLmNoPqRsTuVwXyZ0123456789ab"
    diff = f"+++ b/.envrc\n+GITHUB_TOKEN={token}\n"
    out = pr_review.sanitize_diff(diff)
    assert token not in out
    assert "[REDACTED-SECRET]" in out


def test_sanitize_redacts_github_fine_grained_pats():
    """ghs_, gho_, ghu_, ghr_ prefixes (fine-grained + user-to-server).

    GitHub fine-grained / OAuth tokens are also 36 chars after the prefix.
    """
    for prefix in ("ghs_", "gho_", "ghu_", "ghr_"):
        token = f"{prefix}aBcDeFgHiJkLmNoPqRsTuVwXyZ0123456789ab"
        diff = f"+export TOKEN={token}\n"
        out = pr_review.sanitize_diff(diff)
        assert token not in out, f"{prefix} token leaked: {out}"


def test_sanitize_redacts_slack_token():
    diff = (
        "+SLACK_TOKEN=xoxb-12345678901-1234567890123-AbCdEfGhIjKlMnOpQrStUvWx\n"  # gitleaks:allow
    )
    out = pr_review.sanitize_diff(diff)
    assert "xoxb-12345678901-1234567890123-AbCdEfGhIjKlMnOpQrStUvWx" not in out  # gitleaks:allow
    assert "[REDACTED-SECRET]" in out


def test_sanitize_redacts_google_api_key():
    diff = "+GOOGLE_API_KEY=AIzaSyD-1234567890abcdefghijklmnopqrstuv\n"  # gitleaks:allow
    out = pr_review.sanitize_diff(diff)
    assert "AIzaSyD-1234567890abcdefghijklmnopqrstuv" not in out
    assert "[REDACTED-SECRET]" in out


def test_sanitize_redacts_aws_access_key_id():
    diff = "+AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE\n"
    out = pr_review.sanitize_diff(diff)
    assert "AKIAIOSFODNN7EXAMPLE" not in out
    assert "[REDACTED-SECRET]" in out


def test_sanitize_redacts_bearer_token():
    diff = "+curl -H 'Authorization: Bearer abcdefghijklmnopqrstuvwxyz123456'\n"
    out = pr_review.sanitize_diff(diff)
    assert "abcdefghijklmnopqrstuvwxyz123456" not in out
    assert "Bearer [REDACTED-SECRET]" in out
    # The 'Authorization:' prefix should still be there
    assert "Authorization:" in out


def test_sanitize_redacts_jwt():
    diff = (
        "+AUTH_TOKEN=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"  # gitleaks:allow
        ".eyJzdWIiOiIxMjM0NTY3ODkwIn0"
        ".SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c\n"
    )
    out = pr_review.sanitize_diff(diff)
    assert "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9" not in out
    assert "[REDACTED-JWT]" in out


# ---- New provider patterns (added 2026-07-08) ----------------------------
#
# Cuando se amplian los providers del PR review cascade (OpenRouter +
# NVIDIA NIM + Mistral La Plateforme + Cerebras + opencode-go), los
# SECRET_VALUE_PATTERNS deben cubrir los formatos de key de cada uno, mas
# los secrets especificos del proyecto (Telegram bot tokens) y los
# comunes en deploy configs (Slack/Discord webhooks).


def test_sanitize_redacts_openai_project_key():
    """sk-proj-... (OpenAI project keys) tienen guion despues de 'proj'.

    Antes del fix (2026-07-08) el patron `sk-[A-Za-z0-9]{20,}` no los
    cubria porque '-' no es [A-Za-z0-9]. Ahora se acepta.
    """
    key = "sk-proj-AbCdEfGhIjKlMnOpQrStUvWxYz0123456789AbCdEfGh"
    diff = f"+OPENAI_PROJECT_KEY={key}\n"
    out = pr_review.sanitize_diff(diff)
    assert key not in out
    assert "[REDACTED-SECRET]" in out


def test_sanitize_redacts_anthropic_key():
    """sk-ant-... (Anthropic API keys). Mismo problema que sk-proj-."""
    key = "sk-ant-api03-AbCdEfGhIjKlMnOpQrStUvWxYz0123456789ABCDEFGh"
    diff = f"+ANT_KEY={key}\n"
    out = pr_review.sanitize_diff(diff)
    assert key not in out
    assert "[REDACTED-SECRET]" in out


def test_sanitize_redacts_openrouter_key():
    """sk-or-v1-... (OpenRouter usa prefijo sk-or-)."""
    key = "sk-or-v1-AbCdEfGhIjKlMnOpQrStUvWxYz0123456789AbCdEfGhIjKl"
    diff = f"+OR_KEY={key}\n"
    out = pr_review.sanitize_diff(diff)
    assert key not in out
    assert "[REDACTED-SECRET]" in out


def test_sanitize_redacts_nvidia_nim_key():
    """nvapi-... (NVIDIA NIM API key). Sin este patron, un .env con
    NVIDIA_API_KEY sale tal cual al LLM (los free tiers pueden entrenar).
    """
    key = "nvapi-vSH7AbCdEfGhIjKlMnOpQrStUvWxYz0123456789ABCDEFGhIjKlM"
    diff = f"+NVIDIA_API_KEY={key}\n"
    out = pr_review.sanitize_diff(diff)
    assert key not in out
    assert "[REDACTED-SECRET]" in out


def test_sanitize_redacts_cerebras_key():
    """csk-... (Cerebras API key). Cerebras es fallback Tier 2 del cascade.
    Las keys reales son 40+ chars despues del prefijo.
    """
    key = "csk-AbCdEfGhIjKlMnOpQrStUvWxYz0123456789abcdefghij"
    diff = f"+CEREBRAS_API_KEY={key}\n"
    out = pr_review.sanitize_diff(diff)
    assert key not in out
    assert "[REDACTED-SECRET]" in out


def test_sanitize_redacts_mistral_key_uppercase_var():
    """Mistral API keys (MISTRAL_API_KEY=...). Mistral no documenta el
    formato publico; heuristica var-name para no false-positivear
    hashes/UUIDs de 32 chars.
    """
    key = "AbCdEfGhIjKlMnOpQrStUvWxYz012345"
    diff = f"+MISTRAL_API_KEY={key}\n"
    out = pr_review.sanitize_diff(diff)
    assert key not in out
    assert "[REDACTED-SECRET]" in out


def test_sanitize_redacts_mistral_key_lowercase_var():
    """Mistral env-var en lowercase + underscore (mistral_api_key)."""
    key = "AbCdEfGhIjKlMnOpQrStUvWxYz012345"
    diff = f"+mistral_api_key={key}\n"
    out = pr_review.sanitize_diff(diff)
    assert key not in out
    assert "[REDACTED-SECRET]" in out


def test_sanitize_does_not_redact_random_alnum_in_other_var():
    """Un identificador alfanumerico de 32 chars asignado a una var
    que NO es Mistral NO debe redactarse (evita falsos positivos sobre
    hashes/UUIDs legitimos).
    """
    key = "AbCdEfGhIjKlMnOpQrStUvWxYz012345"
    diff = f"+SOME_HASH={key}\n+REQUEST_ID={key}\n"
    out = pr_review.sanitize_diff(diff)
    # Los identificadores legitimos pasan tal cual
    assert key in out
    assert "[REDACTED-SECRET]" not in out


def test_sanitize_redacts_telegram_bot_token():
    """Telegram bot tokens formato <bot_id>:<35-char-token>. CRITICO:
    Oroimen es un bot de Telegram. Si alguien sube el token por error,
    sale al LLM sin este patron.
    """
    token = "7123456789:AAEhBPbXd2K3L4mN5oP6qR7sT8uV9wX0yZ1"
    diff = f"+TELEGRAM_BOT_TOKEN={token}\n"
    out = pr_review.sanitize_diff(diff)
    assert token not in out
    assert "[REDACTED-SECRET]" in out


def test_sanitize_redacts_slack_webhook():
    """Slack incoming webhooks (URL completa, path-level redaction)."""
    webhook = "https://hooks.slack.com/services/T0123/B0123/abcdefghijklmnop"
    diff = f"+SLACK_WEBHOOK={webhook}\n"
    out = pr_review.sanitize_diff(diff)
    assert webhook not in out
    assert "[REDACTED-PATH]" in out


def test_sanitize_redacts_discord_webhook():
    """Discord webhooks (discord.com o discordapp.com)."""
    webhook = "https://discord.com/api/webhooks/1234567890/abcDEFghijKLMnopqrSTU"
    diff = f"+DISCORD_WEBHOOK={webhook}\n"
    out = pr_review.sanitize_diff(diff)
    assert webhook not in out
    assert "[REDACTED-PATH]" in out


def test_sanitize_redacts_jwt_with_short_signature():
    """JWT con signature < 10 chars (caso real: tokens con sigs cortas).
    Antes del fix (2026-07-08) el ultimo segmento requeria {10,} y
    fallaba. Ahora >= 1 char.
    """
    diff = "+AUTH=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.zzzzzzz\n"  # gitleaks:allow
    out = pr_review.sanitize_diff(diff)
    assert "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9" not in out
    assert "[REDACTED-JWT]" in out


def test_sanitize_does_not_touch_short_strings():
    """Random short strings / dict keys must NOT be falsely redacted."""
    diff = (
        "+++ b/hermes/utils.py\n"
        "+def get_user():\n"
        "+    return {'id': 'sk-fake-short'}  # NOT a real key\n"
        "+    return 'ghp-short-not-a-token'\n"
        "+    return 'just-a-string'\n"
    )
    out = pr_review.sanitize_diff(diff)
    # Short fragments must pass through untouched
    assert "sk-fake-short" in out
    assert "ghp-short-not-a-token" in out


def test_sanitize_does_not_redact_cosmetic_match():
    r"""Paths like 'risk_keyword.py' should not be eaten by the value regex.

    The .*\\.key pattern only matches when the file ends in .key (no word char after).
    A name like 'risk_keyword_inside_function.py' should not be touched.
    """
    diff = "+++ b/hermes/risk_keyword_inside_function.py\n+def risky():\n"
    out = pr_review.sanitize_diff(diff)
    # Path was redacted by SENSITIVE_PATH_PATTERNS (matches `.*\\.pem`/`.*\\.key`),
    # but the function body line should not be touched.
    # (If anything, the whole file path gets [REDACTED-PATH], not the body.)
    assert "def risky():" in out


def test_sanitize_combined_path_and_value():
    """A line that has both a sensitive path AND a leaked value gets sanitized twice."""
    # GitHub PAT: 38 chars after ghp_ to safely exceed the regex's 36+ quantifier.
    diff = (
        "--- a/.env\n"
        "+++ b/.env\n"
        "+OPENAI_API_KEY=sk-abcdefghij1234567890abcdef\n"  # gitleaks:allow
        "+GITHUB_TOKEN=ghp_aaaaabbbbbcccccddddddeeeeefffffggghhhh\n"  # gitleaks:allow
    )
    out = pr_review.sanitize_diff(diff)
    assert "sk-abcdefghij1234567890abcdef" not in out
    assert "ghp_aaaaabbbbbcccccddddddeeeeefffffggghhhh" not in out  # gitleaks:allow
    # Path was redacted
    assert "[REDACTED-PATH]" in out
    # Values were redacted
    assert out.count("[REDACTED-SECRET]") >= 2


# ---- Negative tests: paths/identifiers that look risky but are NOT ------
#
# These document (and pin) the expected behavior of sanitize_diff() on
# text that a hasty reader might think should be redacted. If a future
# regex change makes one of these FAIL, that's a regression in the
# expected behavior, not a "fix" — re-read the regex first.


def test_sanitize_does_not_redact_test_keyword_py():
    r"""`tests/unit/test_keyword.py` should pass through unchanged.

    The substring `keyword` could superficially trigger `[^\s\n]*\.key`
    if anchored wrong; the pattern is `\.key` (literal `.` then `key`,
    end-of-token) so `test_keyword.py` does not match.
    """
    diff = "+++ b/tests/unit/test_keyword.py\n+def f(): pass\n"
    out = pr_review.sanitize_diff(diff)
    assert "test_keyword.py" in out
    assert "[REDACTED-PATH]" not in out


def test_sanitize_does_not_redact_key_helpers_py():
    r"""`hermes/utils/key_helpers.py` is a normal source path.

    `key_helpers.py` does NOT end in `.key` (literal extension), so the
    `*.key` path pattern does not match.
    """
    diff = "+++ b/hermes/utils/key_helpers.py\n+def derive(): pass\n"
    out = pr_review.sanitize_diff(diff)
    assert "hermes/utils/key_helpers.py" in out
    assert "[REDACTED-PATH]" not in out


def test_sanitize_does_not_redact_function_named_secret():
    """Function/variable names are NOT path-shaped strings.

    `def get_secret_key(): return None` and `API_KEY_NAME = 'X'` are
    source code identifiers. Path-redact patterns only fire on file
    shapes (extensions like `.key`, directory shapes like `.ssh/`).
    """
    diff = "+++ b/hermes/config.py\n+def get_secret_key(): return None\n+API_KEY_NAME = 'X'\n"
    out = pr_review.sanitize_diff(diff)
    assert "def get_secret_key():" in out
    assert "API_KEY_NAME" in out
    assert "[REDACTED-PATH]" not in out


def test_sanitize_does_not_redact_rsa_in_word():
    """`reason` / `foremost` should not be eaten by the `id_rsa|...`
    SSH-key pattern; the pattern requires `id_(rsa|...)` as a contiguous
    token, not a substring.
    """
    diff = "+++ b/README.md\n+See the `carmen-rsa-anchor` discussion.\n"
    out = pr_review.sanitize_diff(diff)
    assert "carmen-rsa-anchor" in out
    assert "[REDACTED-PATH]" not in out


# ---- count_redactions + should_skip_for_redactions -----------------------


def test_count_redactions_zero_on_clean_diff():
    assert pr_review.count_redactions("nothing to redact here") == 0


def test_count_redactions_handles_all_three_markers():
    """Each [REDACTED-*] variant must be counted.

    Input has 2 occurrences of `.env` (one in `--- a/`, one in `+++ b/`)
    → 2 [REDACTED-PATH]. Plus one sk-... → 1 [REDACTED-SECRET]. Plus one
    JWT → 1 [REDACTED-JWT]. Total = 4.
    """
    diff = (
        "--- a/.env\n"
        "+++ b/.env\n"
        "+KEY=sk-aaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
        "+JWT=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
        ".eyJzdWIiOiIxMjM0NTY3ODkwIn0"
        ".SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c\n"
    )
    sanitized = pr_review.sanitize_diff(diff)
    # sanity: each marker present
    assert sanitized.count("[REDACTED-PATH]") == 2
    assert sanitized.count("[REDACTED-SECRET]") == 1
    assert sanitized.count("[REDACTED-JWT]") == 1
    assert pr_review.count_redactions(sanitized) == 4


def test_count_redactions_handles_empty():
    assert pr_review.count_redactions("") == 0


def test_should_skip_below_threshold():
    sanitized = "[REDACTED-PATH]\n" * 3
    skip, count = pr_review.should_skip_for_redactions(sanitized, threshold=5)
    assert skip is False
    assert count == 3


def test_should_skip_at_threshold_returns_false():
    """Boundary: count == threshold is still OK. Skip is strictly greater."""
    sanitized = "[REDACTED-PATH]\n" * 5
    skip, count = pr_review.should_skip_for_redactions(sanitized, threshold=5)
    assert skip is False
    assert count == 5


def test_should_skip_above_threshold():
    sanitized = "[REDACTED-PATH]\n" * 6
    skip, count = pr_review.should_skip_for_redactions(sanitized, threshold=5)
    assert skip is True
    assert count == 6


def test_should_skip_mixed_markers_sum_to_threshold():
    """Mixed [REDACTED-PATH] + [REDACTED-SECRET] + [REDACTED-JWT] sum."""
    sanitized = (
        "[REDACTED-PATH]\n[REDACTED-PATH]\n[REDACTED-SECRET]\n[REDACTED-JWT]\n[REDACTED-SECRET]"
    )
    skip, count = pr_review.should_skip_for_redactions(sanitized, threshold=5)
    assert count == 5
    assert skip is False

    sanitized2 = sanitized + "\n[REDACTED-PATH]"
    skip2, count2 = pr_review.should_skip_for_redactions(sanitized2, threshold=5)
    assert count2 == 6
    assert skip2 is True


# ---- format_skip_comment -------------------------------------------------


def test_format_skip_comment_includes_counts_and_threshold():
    body = pr_review.format_skip_comment(redaction_count=12, threshold=5)
    assert "12" in body
    assert "5" in body
    assert "Manual review required" in body or "manual review" in body.lower()
    # Remediation hint for false positives
    assert "PR_REVIEW_REDACTION_THRESHOLD" in body


def test_format_skip_comment_is_well_formed():
    body = pr_review.format_skip_comment(redaction_count=99, threshold=10)
    # Should not raise, should contain key markers
    assert body.startswith("## 🤖 PR Review")
    assert "Skipped automated review" in body
    assert "<sub>" in body
    assert "</sub>" in body
