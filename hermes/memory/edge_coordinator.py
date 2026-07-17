"""EdgeCoordinator — Sprint 19 Slice 4c (§4.4).

Monitors edge PC availability and auto-queues low-confidence Tesseract
results to the edge when a PC is online. Three responsibilities:

1. **Probe loop** (background task): periodically HTTP-probes every
   configured edge PC. Tracks online/offline state per host.

2. **Catch-up pass** (one-shot): when a PC transitions offline→online,
   drains `ocr_pending.status='pending_review'` JPG/PNG rows into the
   edge queue. See TDD §4.4.4 for the "I was on travel for 2 weeks"
   use case.

3. **Enqueue** (sync, called by drop_watcher): writes `request.json`
   to `edge_queue/<file_id>/` and updates `ocr_pending.status=
   'edge_queued'`. Path-only delivery (no `input.bin` copy).

NORTH STAR: Tesseract NAS is the default. Edge PC is FREE local compute
that may give better OCR. Hosted VLM is NEVER auto-invoked from this
coordinator — it requires explicit `/externalOCR` user command (Slice 4d).

PRIVACY: the request.json contains ONLY the POSIX-relative path to the
file. The file content is NEVER copied to the edge queue (Gemini 3.1 Pro
anti-pattern fix: avoids duplicating storage on the NAS + saturating the
SMB share when many files accumulate during travel).

Concurrency model:
- One probe task per coordinator (cheap, single HTTP request per PC).
- Enqueue is per-call from drop_watcher, awaits the file_id write.
- Catch-up runs in the probe task context (waits for batch_delay_ms
  between enqueues to avoid SMB share hammering).

Configuration (env, see hermes/config.py + .env.example):
- EDGE_COMPUTERS: CSV of "hostname:port" or just hostname. Default port 8080.
- OCR_AUTO_EDGE_OCR: master switch. If false, coordinator still probes
  but `enqueue()` always returns False (drop_watcher routes to
  `pending_review` only).
- OCR_EDGE_MAX_QUEUE_SIZE: cap per catch-up pass.
- OCR_EDGE_BATCH_DELAY_MS: debounce between catch-up enqueues.
- OCR_EDGE_AUTOQUEUE_THRESHOLD: confidence cutoff. Below this = queue.
- EDGE_PROBE_TIMEOUT_S, EDGE_PROBE_INTERVAL_S: probe tuning.

Testability:
- `probe_fn` is injectable. Tests pass a mock that returns a list of
  online hostnames; coordinator uses it instead of real HTTP probes.
- `enqueue()` writes to `edge_root` (Path); tests pass a tmp_path.
- All async methods are pure async (no fire-and-forget) for predictable
  test cleanup.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hermes.memory.db import Database
    from hermes.memory.ocr_pending_repo import OcrPendingRepo

logger = logging.getLogger(__name__)


# Queue schema version (TDD §4.4.2). Bump if the request.json shape changes.
REQUEST_SCHEMA_VERSION = 1


def _utc_now_iso() -> str:
    """ISO 8601 UTC timestamp, second precision. Format: YYYY-MM-DDTHH:MM:SSZ."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%Z")


def _utc_now_iso_ms() -> str:
    """ISO 8601 UTC with millisecond precision (for edge_queued_at audit)."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _parse_iso(s: str) -> datetime:
    """Parse an ISO 8601 UTC timestamp. Supports both with and without
    millisecond precision. Returns UTC-aware datetime (tzinfo=UTC).

    Used by M6 Phase 5 to compute the age of a zombie row. We use
    `datetime.strptime` which produces a NAIVE datetime, then we
    attach UTC tzinfo so the result can be compared with `datetime.now(UTC)`.
    """
    # Strip trailing Z and try both with/without millis
    s = s.rstrip("Z")
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=UTC)
        except ValueError:
            continue
    raise ValueError(f"unparseable ISO timestamp: {s!r}")


# Probe function signature: given a list of (hostname, port) tuples,
# return a set of hostnames that are currently online.
# In production: real HTTP GET to http://{host}:{port}/health.
# In tests: mock returns a fixed set.
ProbeFn = Callable[[list[tuple[str, int]]], Awaitable[set[str]]]


@dataclass(frozen=True, slots=True)
class EdgePC:
    """Snapshot of one edge PC's state at probe time."""

    hostname: str
    port: int
    is_online: bool
    last_probe_at: float  # monotonic seconds (time.monotonic())
    last_state_change_at: float  # when the boolean flipped
    latency_ms: float | None = None  # None if offline / probe failed


@dataclass
class _PCState:
    """Mutable per-PC state held by the coordinator. Not exposed."""

    hostname: str
    port: int
    is_online: bool = False
    last_probe_at: float = 0.0
    last_state_change_at: float = 0.0
    latency_ms: float | None = None


class EdgeCoordinator:
    """Monitors edge PCs and auto-queues low-confidence OCR jobs.

    Lifecycle:
        coord = EdgeCoordinator(...)
        await coord.start()        # spawns probe task
        # ... drop_watcher calls coord.enqueue() per file ...
        # ... catch-up happens automatically on offline→online ...
        await coord.stop()         # cancels probe task, awaits cleanup

    The coordinator is safe to use WITHOUT start() — enqueue() and
    is_online() work synchronously based on the last probe result. start()
    is only needed to keep the probe loop running.
    """

    # Default port for the PC daemon's health endpoint. The PC daemon
    # (Sprint 22+ scope) will expose /health on this port.
    DEFAULT_PORT = 8080

    def __init__(
        self,
        *,
        edge_computers: list[tuple[str, int]],
        db: Database,
        ocr_repo: OcrPendingRepo,
        edge_root: Path,
        auto_ocr: bool = True,
        autoqueue_threshold: float = 0.85,
        max_queue_size: int = 1000,
        batch_delay_ms: int = 500,
        probe_timeout_s: float = 1.0,
        probe_interval_s: float = 30.0,
        smb_root_prefix: str = "/mnt/shared/",
        probe_fn: ProbeFn | None = None,
    ) -> None:
        # Defensive copy of the list of (hostname, port) tuples.
        self._pcs: dict[str, _PCState] = {
            h: _PCState(hostname=h, port=p) for h, p in edge_computers
        }
        self._db = db
        self._ocr_repo = ocr_repo
        self._edge_root = edge_root
        self._auto_ocr = auto_ocr
        self._autoqueue_threshold = autoqueue_threshold
        self._max_queue_size = max_queue_size
        self._batch_delay_ms = batch_delay_ms
        self._probe_timeout_s = probe_timeout_s
        self._probe_interval_s = probe_interval_s
        # If no probe_fn is provided, fall back to a real HTTP probe using
        # httpx. Importing httpx at module load is heavy; do it lazily.
        self._probe_fn: ProbeFn = probe_fn or self._default_probe_fn

        # Probe task control
        self._probe_task: asyncio.Task[None] | None = None
        # Catch-up tasks (one per offline→online transition). Held so we
        # can cancel them on stop() and so they don't get GC'd mid-flight
        # ("Task was destroyed but it is pending" warning).
        self._catchup_tasks: set[asyncio.Task[None]] = set()
        self._stop_event: asyncio.Event = asyncio.Event()

        # Path-only delivery: strip the deployment-specific shared-root
        # prefix from vault_files.path. The public default is generic; private
        # deployments override EDGE_SMB_ROOT_PREFIX in their uncommitted .env.
        normalized_prefix = smb_root_prefix.strip().replace("\\", "/")
        if not normalized_prefix:
            raise ValueError("smb_root_prefix must not be empty")
        self._smb_root_prefix = normalized_prefix.rstrip("/") + "/"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Spawn the periodic probe loop. Idempotent: if already running,
        no-op. The first probe happens immediately; subsequent probes
        every `probe_interval_s` seconds.
        """
        if self._probe_task is not None and not self._probe_task.done():
            return
        self._stop_event.clear()
        self._probe_task = asyncio.create_task(
            self._probe_loop(),
            name="edge_coordinator_probe",
        )
        logger.info(
            "edge_coordinator_started",
            extra={"pc_count": len(self._pcs), "interval_s": self._probe_interval_s},
        )

    async def stop(self) -> None:
        """Cancel the probe task. Awaits in-flight probe to finish. Safe
        to call multiple times.
        """
        self._stop_event.set()
        if self._probe_task is not None:
            self._probe_task.cancel()
            # Await the task; the CancelledError is the expected outcome
            # of .cancel(). We intentionally suppress it because the
            # cancellation is OUR action, not an error.
            try:  # noqa: SIM105
                await self._probe_task
            except (asyncio.CancelledError, BaseException):
                pass
            self._probe_task = None
        # Cancel any in-flight catch-up tasks. They check the stop_event
        # and the is_online() check inside their loop, so they may exit
        # cleanly even before cancellation propagates.
        for task in list(self._catchup_tasks):
            task.cancel()
        for task in list(self._catchup_tasks):
            try:  # noqa: SIM105
                await task
            except (asyncio.CancelledError, BaseException):
                pass
        self._catchup_tasks.clear()
        logger.info("edge_coordinator_stopped")

    async def is_online(self) -> bool:
        """True if at least one configured PC is online. Cheap O(N)."""
        return any(s.is_online for s in self._pcs.values())

    async def get_state(self) -> list[EdgePC]:
        """Snapshot of all configured PCs' current state. For telemetry
        + tests.
        """
        return [
            EdgePC(
                hostname=s.hostname,
                port=s.port,
                is_online=s.is_online,
                last_probe_at=s.last_probe_at,
                last_state_change_at=s.last_state_change_at,
                latency_ms=s.latency_ms,
            )
            for s in self._pcs.values()
        ]

    async def enqueue(
        self,
        *,
        file_id: str,
        path: str,
        local_confidence: float,
        requested_by: str = "auto-queue",
    ) -> bool:
        """Enqueue a file to the edge queue. Returns True on success.

        Returns False (does not raise) when:
        - auto_ocr is False (master switch off)
        - confidence >= autoqueue_threshold (Tesseract result is good enough)
        - no PC is online
        - filesystem write fails (logs error, returns False)

        On success: writes request.json to edge_queue/<file_id>/ AND
        updates ocr_pending status='edge_queued' with edge_model + edge_queued_at.

        Idempotent: if the file is already in edge_queued, returns True
        without re-writing (the existing request.json is kept).
        """
        if not self._auto_ocr:
            return False
        if local_confidence >= self._autoqueue_threshold:
            return False
        if not await self.is_online():
            return False

        # Idempotency check: if already edge_queued, do nothing.
        existing = await self._ocr_repo.get(file_id)
        if existing is not None and existing.status == "edge_queued":
            return True

        # Write request.json (path-only delivery per TDD §4.4.2)
        try:
            smb_relative = self._to_smb_relative(path)
            request = {
                "schema_version": REQUEST_SCHEMA_VERSION,
                "file_id": file_id,
                "path": smb_relative,
                "original_path": path,
                "local_confidence": local_confidence,
                "requested_at": _utc_now_iso(),
                "requested_by": requested_by,
                "queue_source": "tesseract_local",
            }
            # Run sync filesystem I/O in a worker thread to avoid
            # blocking the event loop (Sprint 19 LLM review 2026-07-11).
            # mkdir + write_text is a quick op on the local NAS, but in
            # catch-up scenarios (1000 files) it adds up.
            request_path = await asyncio.to_thread(self._write_request_json, file_id, request)
        except OSError as exc:
            logger.error(
                "edge_enqueue_filesystem_error",
                extra={"file_id": file_id, "error": str(exc)},
            )
            return False

        # Update ocr_pending row
        updated = await self._ocr_repo.update_status(
            file_id,
            "edge_queued",
            edge_model="pc-pending",  # PC daemon will overwrite on response
            edge_queued_at=_utc_now_iso_ms(),
        )
        if not updated:
            # Row didn't exist. drop_watcher should have created it.
            # Roll back the filesystem write to avoid orphan request.json.
            with suppress(OSError):
                await asyncio.to_thread(request_path.unlink)
            logger.warning(
                "edge_enqueue_ocr_row_missing",
                extra={"file_id": file_id},
            )
            return False

        logger.info(
            "edge_enqueue_ok",
            extra={
                "file_id": file_id,
                "local_confidence": local_confidence,
                "requested_by": requested_by,
            },
        )
        return True

    # ------------------------------------------------------------------
    # M6 Phase 5: zombie edge job recovery (TDD §4.4.5)
    # ------------------------------------------------------------------

    async def recover_zombies(self, timeout_hours: int) -> int:
        """Reset edge_queued rows older than `timeout_hours` to
        `pending_review`, AND clean up the corresponding orphan
        `request.json` in `edge_queue/<file_id>/`.

        This is M6 Phase 5 (Sprint 19 §4.4.5). It handles the case where
        a PC crashed mid-processing and never wrote a `response.json`,
        leaving the `ocr_pending` row stuck in `edge_queued` forever.

        The partial index `idx_ocr_pending_edge_queued_at` makes the
        SELECT cheap even when `ocr_pending` has millions of rows.

        Args:
            timeout_hours: rows with `edge_queued_at` older than NOW()
                minus this many hours are considered zombies. Default 2h
                per TDD (10x worst-case LLaVA batch time).

        Returns: count of rows recovered. Safe to call repeatedly
        (idempotent: rows already in `pending_review` are skipped by
        the SQL WHERE clause).
        """
        cutoff = datetime.now(UTC) - timedelta(hours=timeout_hours)
        cutoff_iso = cutoff.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        zombies = await self._ocr_repo.fetch_zombie_candidates(cutoff_iso)
        if not zombies:
            return 0
        recovered = 0
        for row in zombies:
            reverted = await self._ocr_repo.revert_to_pending(row.file_id)
            if not reverted:
                # Race: someone else already reverted (manual /edgeOCR retry
                # or another Phase 5 instance). Skip filesystem cleanup too.
                continue
            # Clean up orphan request.json. Best-effort: if the file is
            # already gone (e.g. PC processed it but never wrote response),
            # unlink() raises FileNotFoundError; we swallow.
            request_path = self._edge_root / "edge_queue" / row.file_id / "request.json"
            with suppress(OSError):
                request_path.unlink()
            logger.info(
                "edge_zombie_recovered",
                extra={
                    "file_id": row.file_id,
                    "queued_at": row.edge_queued_at,
                    "age_hours": (
                        (datetime.now(UTC) - _parse_iso(row.edge_queued_at)).total_seconds() / 3600
                        if row.edge_queued_at
                        else None
                    ),
                },
            )
            recovered += 1
        logger.info(
            "edge_zombie_recovery_complete",
            extra={"recovered": recovered, "timeout_hours": timeout_hours},
        )
        return recovered

    # ------------------------------------------------------------------
    # Probe loop + catch-up
    # ------------------------------------------------------------------

    async def _probe_loop(self) -> None:
        """Periodic probe. Runs until stop() is called or task is cancelled."""
        while not self._stop_event.is_set():
            try:
                await self._probe_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("edge_probe_loop_error")
            # Sleep until next probe (interruptible by stop_event)
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=self._probe_interval_s,
                )
                # If we get here, stop_event was set; loop exits.
                break
            except TimeoutError:
                # Normal: interval elapsed, do another probe.
                pass

    async def _probe_once(self) -> None:
        """One round of probes for all configured PCs. Triggers catch-up
        on offline→online transitions.
        """
        if not self._pcs:
            return
        targets = [(s.hostname, s.port) for s in self._pcs.values()]
        online_set = await self._probe_fn(targets)
        now = time.monotonic()
        for s in self._pcs.values():
            was_online = s.is_online
            is_online = s.hostname in online_set
            s.is_online = is_online
            s.last_probe_at = now
            if is_online != was_online:
                s.last_state_change_at = now
                if is_online:
                    logger.info(
                        "edge_pc_online",
                        extra={"hostname": s.hostname, "port": s.port},
                    )
                    # Fire catch-up (don't await if it would block the probe loop).
                    # Track the task so we can cancel it on stop() and prevent
                    # "Task was destroyed but it is pending" warnings. The
                    # done callback removes it from the set on completion.
                    task = asyncio.create_task(
                        self._catchup_pass(),
                        name=f"edge_catchup_{s.hostname}",
                    )
                    self._catchup_tasks.add(task)
                    task.add_done_callback(self._catchup_tasks.discard)
                else:
                    logger.info(
                        "edge_pc_offline",
                        extra={"hostname": s.hostname, "port": s.port},
                    )

    async def _catchup_pass(self) -> None:
        """Drain pending_review JPG/PNG rows to the edge queue.

        Debounced with `_batch_delay_ms` between enqueues to avoid
        hammering the SMB share. Capped at `_max_queue_size` files
        (overflow stays in `pending_review` for the next reconnect).
        """
        extensions = [".jpg", ".jpeg", ".png"]
        rows = await self._ocr_repo.fetch_pending_for_catchup(
            path_extensions=extensions,
            limit=self._max_queue_size,
        )
        logger.info(
            "edge_catchup_start",
            extra={"candidate_count": len(rows), "limit": self._max_queue_size},
        )
        # Use enumerate for O(1) index (was previously rows.index() = O(n^2)
        # inside the loop). Sprint 19 LLM review 2026-07-11.
        for idx, (file_id, path, local_confidence) in enumerate(rows):
            # Re-check online state — PC may have gone offline mid-drain
            if not await self.is_online():
                logger.warning(
                    "edge_catchup_aborted_pc_offline",
                    extra={"drained": idx},
                )
                break
            ok = await self.enqueue(
                file_id=file_id,
                path=path,
                local_confidence=local_confidence,
                requested_by="auto-catchup",
            )
            if not ok:
                # Already covered by enqueue() logging
                continue
            await asyncio.sleep(self._batch_delay_ms / 1000.0)
        logger.info(
            "edge_catchup_done",
            extra={"processed": len(rows)},
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _to_smb_relative(self, vault_path: str) -> str:
        """Strip the SMB root prefix from a vault_files.path to get the
        POSIX-relative path the PC expects.

        Example: /mnt/shared/Documentos/_inbox/foo.jpg -> Documentos/_inbox/foo.jpg

        If the path doesn't start with the prefix, return it as-is
        (defensive: the PC daemon will reject it later with a clear error).
        """
        if vault_path.startswith(self._smb_root_prefix):
            return vault_path[len(self._smb_root_prefix) :]
        return vault_path

    def _write_request_json(self, file_id: str, request: dict) -> Path:
        """Sync helper: write request.json to edge_queue/<file_id>/.

        Returns the path. Called via asyncio.to_thread from enqueue()
        to avoid blocking the event loop on filesystem I/O.

        Why a separate method: easier to mock in tests, and keeps
        enqueue()'s async flow readable.
        """
        queue_dir = self._edge_root / "edge_queue" / file_id
        queue_dir.mkdir(parents=True, exist_ok=True)
        request_path = queue_dir / "request.json"
        request_path.write_text(
            json.dumps(request, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return request_path

    async def _default_probe_fn(self, targets: list[tuple[str, int]]) -> set[str]:
        """Default HTTP probe: GET http://{host}:{port}/health with timeout.

        Returns the set of hostnames that responded with 2xx.
        PC is considered online if it returns ANY 2xx (even if the
        daemon isn't fully ready — the catch-up will retry on next
        probe cycle if the daemon rejects the request).

        If httpx is not available (dev env without it), returns empty
        set (no PCs online) — the system degrades gracefully to
        `pending_review`-only mode.
        """
        try:
            import httpx  # local import: optional dep
        except ImportError:
            logger.warning("edge_probe_httpx_missing")
            return set()
        online: set[str] = set()
        async with httpx.AsyncClient(timeout=self._probe_timeout_s) as client:
            for host, port in targets:
                url = f"http://{host}:{port}/health"
                try:
                    resp = await client.get(url)
                    if 200 <= resp.status_code < 300:
                        online.add(host)
                except Exception:
                    # Timeout, connection refused, DNS failure — all = offline
                    pass
        return online
