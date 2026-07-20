"""S10.4: Telegram push notifications para health alerts.

Sprint 11 (WebUI-primary + Pixel STT) depende de S10.4 porque
WebUI es single point of failure. Si WebUI cae, el user necesita
saber via push notification (Telegram Bot API).

Scope (TDD S10.4 v1.3):
- Solo health alerts: API down/up, backup failed, disk low, db error.
- No incluye Deep Research completion (S10B) ni Obsidian file watcher (S10.3).
- Deduplicacion 1/hora por alert_type para no spamear al user.

Setup requerido (manual, una vez):
1. Crear bot con @BotFather en Telegram, obtener token.
2. Anadir TELEGRAM_BOT_TOKEN al .env de hermes (ya existe para
   el polling de Telegram, se reutiliza).
3. User inicia conversacion con el bot, anota su chat_id (el bot
   lo recibe en update; el script setup_chat_id.py ayuda).
4. Set TELEGRAM_CHAT_ID en .env.
"""

from __future__ import annotations

import logging
import os
import time
from typing import ClassVar

logger = logging.getLogger(__name__)


# Telegram Bot API endpoint
_TELEGRAM_API = "https://api.telegram.org/bot{token}/{method}"


class TelegramNotifier:
    """Push notifications via Telegram Bot API (S10.4).

    Decoupled from the polling bot (TelegramBot in receivers/polling).
    This notifier is for OUTBOUND health alerts from Oroimen → user.
    The polling bot is for INBOUND messages from user → Oroimen.

    Deduplication: 1 alert per (alert_type) per cooldown window
    (default 3600s = 1h). Prevents spam when a check fails repeatedly.
    """

    # Iconos por severity. Orden importa para tests.
    # ClassVar: dict es compartido entre instancias, no deberia ser
    # instance attribute. Los emojis ℹ ⚠ 🚨 son intencionales
    # (RUF001/RUF003 silenciados via [lint.per-file-ignores] en
    # ruff.toml; ver comentario alli).
    _ICONS: ClassVar[dict[str, str]] = {
        "info": "ℹ️",
        "warning": "⚠️",
        "critical": "🚨",
    }

    def __init__(
        self,
        bot_token: str | None = None,
        chat_id: str | None = None,
        cooldown_seconds: int = 3600,
    ) -> None:
        # Si no se pasan, intentar leer de env. Si no estan, no-op
        # graceful (log warning, no crash).
        self._bot_token = bot_token or os.environ.get("TELEGRAM_BOT_TOKEN")
        self._chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID")
        self._cooldown_seconds = cooldown_seconds
        # alert_type -> unix timestamp del ultimo envio
        self._last_sent: dict[str, float] = {}
        if not self._bot_token or not self._chat_id:
            logger.warning(
                "telegram_notifier_disabled",
                extra={
                    "reason": "missing_token_or_chat_id",
                    "has_token": bool(self._bot_token),
                    "has_chat_id": bool(self._chat_id),
                },
            )
            self._enabled = False
        else:
            self._enabled = True

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _should_send(self, alert_type: str) -> bool:
        """Check dedup window. Returns True if alert should be sent."""
        if not self._enabled:
            return False
        now = time.time()
        last = self._last_sent.get(alert_type, 0.0)
        if now - last < self._cooldown_seconds:
            logger.debug(
                "telegram_alert_suppressed_cooldown",
                extra={"alert_type": alert_type, "since_last_seconds": now - last},
            )
            return False
        self._last_sent[alert_type] = now
        return True

    async def send_health_alert(
        self,
        alert_type: str,
        message: str,
        severity: str = "warning",
    ) -> bool:
        """Envia health alert via Telegram. Returns True si se envio.

        Args:
            alert_type: 'api_down' | 'api_recovered' | 'backup_failed' |
                'disk_low' | 'db_error' | 'process_dead'.
            message: texto del alert (max 4096 chars para Telegram).
            severity: 'info' | 'warning' | 'critical'.

        Returns:
            True si el mensaje se envio (HTTP 200 de Telegram).
            False si fue suppressed por cooldown, no enabled, o error.
        """
        if alert_type not in self._ICONS:
            logger.warning(
                "telegram_alert_unknown_type",
                extra={"alert_type": alert_type, "valid_types": list(self._ICONS.keys())},
            )
        if not self._should_send(alert_type):
            return False
        icon = self._ICONS.get(severity, "ℹ️")
        text = f"{icon} Oroimen Health Alert\n\n`{alert_type}`: {message}"
        return await self._send_telegram(text)

    async def _send_telegram(self, text: str) -> bool:
        """POST a Telegram Bot API sendMessage.

        Markdown fallback (TDD §9.4): si sendMessage con parse_mode=MARKDOWN
        falla por error de parse/entity (e.g. underscores conflictivos con
        'abc_xyz'), reintenta como plain text. Esto evita perder el push
        por un caracter raro en el contenido.
        """
        if not self._enabled:
            return False
        try:
            import httpx

            url = _TELEGRAM_API.format(token=self._bot_token, method="sendMessage")
            async with httpx.AsyncClient(timeout=10.0) as client:
                # Attempt 1: Markdown
                r = await client.post(
                    url,
                    json={
                        "chat_id": self._chat_id,
                        "text": text,
                        "parse_mode": "Markdown",
                    },
                )
            if r.status_code == 200:
                logger.info("telegram_alert_sent", extra={"status": r.status_code})
                return True

            # Markdown parse fallback (TDD §9.4): si el error es de parse/entity
            body = r.text or ""
            is_parse_error = (
                "can't parse" in body.lower() or "entity" in body.lower() or "parse" in body.lower()
            )
            if r.status_code == 400 and is_parse_error:
                logger.warning(
                    "telegram_markdown_fallback",
                    extra={"status": r.status_code, "body": body[:200]},
                )
                # Attempt 2: plain text
                try:
                    async with httpx.AsyncClient(timeout=10.0) as client:
                        r2 = await client.post(
                            url,
                            json={"chat_id": self._chat_id, "text": text},
                        )
                    if r2.status_code == 200:
                        logger.info(
                            "telegram_alert_sent_plain",
                            extra={"status": r2.status_code},
                        )
                        return True
                    logger.warning(
                        "telegram_alert_plain_failed",
                        extra={"status": r2.status_code, "body": r2.text[:200]},
                    )
                    return False
                except Exception as exc2:
                    logger.exception(
                        "telegram_alert_plain_error",
                        extra={"error_type": type(exc2).__name__},
                    )
                    return False

            logger.warning(
                "telegram_alert_failed",
                extra={"status": r.status_code, "body": body[:200]},
            )
            return False
        except Exception as exc:
            logger.exception(
                "telegram_alert_error",
                extra={"error_type": type(exc).__name__},
            )
            return False

    # ============================================================
    # Sprint 14 (TDD_S14_DEEP_RESEARCH.md §9): research job notifications
    # ============================================================

    async def send_research_complete(
        self,
        job_id: str,
        cost_usd: float,
    ) -> bool:
        """Push al user cuando un research job completa exitosamente.

        Slice 1C2: the signature is now ``(job_id, cost_usd)`` — no
        ``output_path`` / ``report_ref`` parameter. The template is
        redacted to remove the filesystem path: clients retrieve the
        report content via ``GET /v1/jobs/{id}/report`` (which is
        auth-gated, owner-scoped, and path-confined). The Telegram
        message is now a delivery notice only.

        Cooldown: 1h por job_id (NO por alert_type, cada job tiene 1 notif).
        Implementación: usar alert_type=f"research_complete:{job_id}" para
        granularidad job_id (cada job su propio slot de cooldown).

        Format del mensaje (Slice 1C2, owner-adjudicated):
            ✅ Research complete · Job `abc123def456`
            📄 Report ready in Oroimen
            💰 $0.0420
            🌐 Open Oroimen to view it

        Returns:
            True si enviado, False si suppressed (cooldown o not enabled).
        """
        alert_key = f"research_complete:{job_id}"
        if not self._should_send(alert_key):
            return False

        text = (
            f"✅ Research complete · Job `{job_id}`\n"
            f"📄 Report ready in Oroimen\n"
            f"💰 ${cost_usd:.4f}\n"
            f"🌐 Open Oroimen to view it"
        )
        return await self._send_telegram(text)

    async def send_research_failed(
        self,
        job_id: str,
        error_taxonomy: str,
        error_message: str,
        retryable: bool,
    ) -> bool:
        """Push cuando un research job falla tras agotar retries.

        Format (TDD §9.1):
            ❌ Research failed · Job `abc123def456`
            ⚠️ llm_4xx: content_policy_violation
            💡 re-submit via webapp to retry

        NOTA: el comando /retry TG NO se implementa en S14 (deferido a S15
        Could). En S14 el "retry" se hace vía webapp o re-POST /v1/jobs
        con el mismo query.
        """
        alert_key = f"research_failed:{job_id}"
        if not self._should_send(alert_key):
            return False

        action = "re-submit via webapp to retry" if retryable else "no retry — manual fix needed"
        text = (
            f"❌ Research failed · Job `{job_id}`\n"
            f"⚠️ {error_taxonomy}: {error_message}\n"
            f"💡 {action}"
        )
        return await self._send_telegram(text)

    def reset_cooldown(self, alert_type: str | None = None) -> None:
        """Test helper: limpia dedup state. Si alert_type=None, limpia todo."""
        if alert_type is None:
            self._last_sent.clear()
        else:
            self._last_sent.pop(alert_type, None)
