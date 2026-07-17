"""Sprint 9.3: BudgetTracker para Web Search Router.

Tracking de budget mensual por backend (SearXNG unlimited, Tavily
1k/mes, Exa 1k/mes). Tabla `search_budget` en conversations.db
(respaldada por backup S8.4).

Atomicidad: `INSERT OR IGNORE` + `UPDATE used = used + ?` evita
race conditions en requests paralelos (BEGIN IMMEDIATE no es
necesario porque cada operacion es single-statement y SQLite
serializa writes por default).

Month rollover: cada operacion chequea si el row tiene el mes
actual; si no, resetea `used` a 0 y actualiza `month`. Esto es
mas limpio que un cron job separado y no requiere estado en
memoria.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hermes.memory.db import Database

logger = logging.getLogger(__name__)


class BudgetTracker:
    """Budget tracker por backend con month rollover automatico.

    Attributes:
        db: Database instance (usa la misma conexion que el resto).
        limits: dict[backend_name, monthly_limit]. -1 = ilimitado
            (SearXNG self-hosted). 0 = backend no disponible.
    """

    def __init__(self, db: Database, limits: dict[str, int]) -> None:
        self._db = db
        self._limits = limits

    async def has_budget(self, backend: str) -> bool:
        """True si el backend tiene budget disponible este mes.

        Si el row tiene un mes viejo, lo resetea atomicamente primero.
        Si el backend no esta en limits, retorna False (limit=0).
        Si limit=-1 (ilimitado), retorna True siempre.
        """
        await self._reset_if_old_month(backend)
        limit = self._get_limit(backend)
        if limit < 0:
            return True  # unlimited
        if limit == 0:
            return False
        used = await self._get_used(backend)
        return used < limit

    async def record_usage(self, backend: str, count: int = 1) -> None:
        """Registra uso del backend (atomic).

        Hace UPSERT: INSERT OR IGNORE si no existe, luego UPDATE
        used = used + count. Si el mes es viejo, primero resetea.

        Args:
            backend: nombre del backend.
            count: cantidad de uso a registrar (default 1). Si es 0,
                es no-op.
        """
        if count == 0:
            return
        await self._reset_if_old_month(backend)
        month = datetime.now(UTC).strftime("%Y-%m")
        # UPSERT pattern: INSERT si no existe, luego UPDATE.
        # Cada statement es atomica en SQLite, suficiente para
        # incrementar counter sin race conditions.
        await self._db.conn.execute(
            "INSERT OR IGNORE INTO search_budget (month, backend, used) " "VALUES (?, ?, 0)",
            (month, backend),
        )
        await self._db.conn.execute(
            "UPDATE search_budget SET used = used + ? " "WHERE month = ? AND backend = ?",
            (count, month, backend),
        )
        await self._db.conn.commit()

    async def remaining(self, backend: str) -> int:
        """Retorna el budget restante este mes.

        -1 si el backend es ilimitado.
        0 si no hay budget o se excedio.
        """
        limit = self._get_limit(backend)
        if limit < 0:
            return -1
        if limit == 0:
            return 0
        used = await self._get_used(backend)
        return max(0, limit - used)

    # --- Internal helpers ---

    def _get_limit(self, backend: str) -> int:
        """Retorna el limit del backend (0 si no esta configurado)."""
        return self._limits.get(backend, 0)

    async def _get_used(self, backend: str) -> int:
        """Lee el `used` actual del row (asume mes actual)."""
        month = datetime.now(UTC).strftime("%Y-%m")
        async with self._db.conn.execute(
            "SELECT used FROM search_budget " "WHERE month = ? AND backend = ?",
            (month, backend),
        ) as cur:
            row = await cur.fetchone()
        return row[0] if row else 0

    async def _reset_if_old_month(self, backend: str) -> None:
        """Si el row tiene un mes viejo, lo resetea a 0 y actualiza month.

        Llamado al inicio de has_budget y record_usage. Idempotente
        si el mes ya es el actual.
        """
        current_month = datetime.now(UTC).strftime("%Y-%m")
        # UPDATE solo afecta si el mes NO es el actual.
        # El WHERE clause usa current_month, asi que solo actualiza
        # rows con meses diferentes (los rows del mes actual no se
        # tocan porque su month != 'old_month' no se cumple).
        async with self._db.conn.execute(
            "UPDATE search_budget "
            "SET used = 0, month = ?, last_reset_at = CURRENT_TIMESTAMP "
            "WHERE backend = ? AND month != ?",
            (current_month, backend, current_month),
        ) as cur:
            updated = cur.rowcount
        if updated > 0:
            await self._db.conn.commit()
            logger.info(
                "search_budget_month_rollover",
                extra={"backend": backend, "new_month": current_month},
            )
