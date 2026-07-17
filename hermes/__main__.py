"""Punto de entrada: python -m hermes."""

import asyncio
import logging
import sys
import time
from collections.abc import Awaitable, Callable
from contextlib import suppress
from typing import Any

from hermes.config import Settings
from hermes.health import HealthServer
from hermes.logging_setup import configure_logging
from hermes.memory.db import Database
from hermes.receivers.polling import PollingReceiver
from hermes.shutdown import install_signal_handlers
from hermes.telemetry import Telemetry
from hermes.tools.builtin import register_builtin_tools
from hermes.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


def _parse_edge_computers(raw: str) -> list[tuple[str, int]]:
    """Parse EDGE_COMPUTERS env var (CSV of hostnames or host:port).

    Examples:
        "" -> []
        "edge.local" -> [("edge.local", 8080)]
        "edge.local,edge.example.com" -> [("edge.local", 8080), ("edge.example.com", 8080)]
        "edge.local:9000" -> [("edge.local", 9000)]

    Returns list of (hostname, port) tuples. Empty list = coordinator
    is not created (drop_watcher still works, but no auto-queue).

    Per TDD §4.4.4: mDNS discovery is Sprint 22+ scope. For now, only
    explicit hostnames in EDGE_COMPUTERS.
    """
    if not raw or not raw.strip():
        return []
    out: list[tuple[str, int]] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" in token:
            host, _, port_str = token.rpartition(":")
            try:
                port = int(port_str)
            except ValueError:
                logger.warning(
                    "edge_computers_bad_port",
                    extra={"token": token, "port_str": port_str},
                )
                continue
            out.append((host, port))
        else:
            out.append((token, 8080))
    return out


async def _close_core_resources(
    *,
    embeddings_service: Any,
    telemetry: Any,
    llm: Any,
    db: Any,
) -> None:
    """Close every core resource even when an earlier closer fails."""
    closers: list[tuple[str, Callable[[], Awaitable[None]]]] = [
        ("embeddings", embeddings_service.aclose),
        ("telemetry", telemetry.aclose),
        ("llm", llm.aclose),
        ("db", db.close),
    ]
    first_error: Exception | None = None
    for name, closer in closers:
        try:
            await closer()
        except Exception as exc:
            logger.exception("core_resource_close_failed", extra={"resource": name})
            if first_error is None:
                first_error = exc
    if first_error is not None:
        raise first_error


async def run() -> int:
    settings = Settings()
    configure_logging(settings.log_level)

    logger.info("hermes_starting", extra={"version": "0.1.0"})

    # PR #118 (Sprint 18 hardening, Gemini P0 #2 "Peligro de Apagado"):
    # Initialize the GLOBAL stop_event FIRST so the EmbedWatcher can be
    # wired with it later in this function. The previous order
    # (stop_event defined at line 509, watcher wired at line 234) made
    # the watcher create its OWN internal stop_event that Hermes
    # shutdown never observed — so SIGTERM didn't propagate and the
    # watcher blocked shutdown for the full LLM timeout.
    stop_event = asyncio.Event()
    install_signal_handlers(stop_event)

    # Sprint 11 (ADR-004): deprecation warning. Sprint 11.0 mantiene
    # enable_telegram=True por default (no romper deployments existentes).
    # Sprint 12+ migrara a default False. Mientras tanto, este warning
    # indica a operators que el bot Telegram sera opt-in legacy en la
    # proxima major release. Para apagarlo: ENABLE_TELEGRAM=false en .env.
    #
    # NOTA: el campo se llama `note` (no `message`) porque `message` es
    # un campo reservado de Python logging — sobreescribirlo via
    # `extra=` lanza KeyError. Ver el fixture `restore_root_logger`
    # en test_main.py para mas detalle.
    if settings.enable_telegram:
        logger.warning(
            "s11_telegram_deprecation_notice",
            extra={
                "note": "Telegram bot es opt-in legacy desde Sprint 11 "
                "(ADR-004). Default actual: True. Sprint 12+: default = False. "
                "Para migrar a WebUI-primary: ENABLE_TELEGRAM=false, "
                "ENABLE_HTTP_API=true (este ya es default).",
            },
        )

    # Sprint 4 MVP-2: setup idempotente de Agent-Reach (config dir,
    # yt-dlp JS runtime, Exa MCP). Si las deps no estan instaladas,
    # se loggea warning y se continua (las tools daran error user-friendly
    # cuando se invoquen).
    if settings.outbound_tools_enabled:
        from scripts.setup_agent_reach import setup_agent_reach

        setup_agent_reach()

    db = Database(settings.db_path)
    # ============================================================================
    # CRÍTICO: db.initialize() DEBE ejecutarse ANTES de iniciar uvicorn.
    # ============================================================================
    # Si la migración corre DESPUÉS de que el server acepte tráfico, los
    # primeros requests pueden fallar con "no such table" o "no such column"
    # (schema race condition). El test test_migration_timing.py detecta esto.
    # NO MOVER las líneas de db.initialize() debajo del bloque del
    # servidor HTTP (ver `if settings.enable_http_api:` más abajo).
    # ============================================================================
    await db.initialize()

    # Sprint 14/17 (PR #113c, B3 fix): init_influx() es la única vía
    # para activar `write_research_metric`. Si InfluxDB no está
    # configurado (INFLUXDB_URL/TOKEN no en env), el init es un
    # no-op graceful y los writes son silenciosos. Llamamos ANTES de
    # cualquier path que pueda emitir métricas (vault scheduler,
    # embed watcher, etc.) para que el primer tick no se pierda.
    from hermes.observability.influxdb import init_influx

    init_influx()

    # Sprint 19 Slice 6: PARA seeding (TDD §5 lines 1355-1379).
    # After DB open + before schedulers. Idempotent: re-running is a
    # no-op. Creates 4 default collections (01_Proyectos_Activos,
    # 02_Áreas_de_Responsabilidad, 03_Recursos_y_Conocimiento,
    # 04_Archivo) on first startup. Fail-fast if DB can't accept
    # the seed — Hermes can't operate without the PARA defaults.
    # Sprint 19 Slice 4d v2 (commit 3): seed_all is the orchestrator
    # that calls seed_para_collections + seed_inbox_collections.
    from hermes.memory.collections import VaultCollectionsRepo
    from hermes.memory.seed import migrate_legacy_para_names, seed_all

    vault_collections_repo = VaultCollectionsRepo(db)
    # One-shot migration: rename legacy accented PARA names to ASCII.
    # Idempotent; safe on every startup.
    await migrate_legacy_para_names(vault_collections_repo)
    # Seed all defaults (PARA + _inbox per monitor root).
    created = await seed_all(settings, vault_collections_repo)
    logger.info(
        "vault_collections_seeded",
        extra={"created_count": created},
    )

    # Sprint 8 S8.4: backup scheduler (WAL online, daily 03:30 AM).
    # Opt out with BACKUP_ENABLED=false. Export files from backup_dir using
    # the backup system appropriate for the deployment.
    backup_scheduler = None
    if settings.backup_enabled:
        from hermes.scheduler import BackupScheduler

        backup_scheduler = BackupScheduler(
            hour=settings.backup_hour,
            minute=settings.backup_minute,
        )
        await backup_scheduler.start()

    telemetry = Telemetry(settings)

    # Sprint 6 T53 v3.1: HealthServer solo se inicia si HTTP API esta
    # desactivado. Si HTTP API esta activo, FastAPI expone su propio
    # /health en el mismo puerto 8000 (evitamos Address already in use).
    health: HealthServer | None = None
    if not settings.enable_http_api:
        health = HealthServer(settings.health_host, settings.health_port, db=db)
        await health.start()

    # Sprint 6 T53 v3.1: LLMRouter se construye SIEMPRE, no solo cuando
    # tools_enabled. Razon: el HTTP API (si esta activo) necesita el
    # router aunque no haya tools. Crearlo siempre simplifica la logica
    # y el coste es bajo (httpx client + circuit breakers).
    from hermes.llm.router import LLMRouter

    llm = LLMRouter(settings)

    # Sprint 9.1: EmbeddingsService (RAG). Si OPENAI_API_KEY no está
    # configurado, opera en modo disabled (chat funciona, RAG no).
    from hermes.services.embeddings import EmbeddingsService

    embeddings_service = EmbeddingsService(settings, db)

    # Sprint 9.2: Sleep Cycle (memory fact extraction, opt-in).
    # Solo se inicia si settings.sleep_cycle_enabled=True. Default False
    # para que el user decida explicitamente. Se crea AQUI (despues de
    # llm y embeddings_service) porque SleepCycle los necesita.
    sleep_cycle_scheduler = None
    if settings.sleep_cycle_enabled:
        from hermes.memory.sleep_cycle import SleepCycle
        from hermes.scheduler import SleepCycleScheduler

        sleep_cycle = SleepCycle(
            db=db,
            router=llm,
            settings=settings,
            embeddings_service=embeddings_service,
        )
        sleep_cycle_scheduler = SleepCycleScheduler(
            hour=settings.sleep_cycle_hour,
            sleep_cycle=sleep_cycle,
        )
        await sleep_cycle_scheduler.start()

    # Sprint 9.4: Conversation cleanup (archive stale, every N min).
    # Opt-out con CLEANUP_ENABLED=false. Default True: el bug 9.3.2b
    # (UNIQUE constraint violation por huérfanas) es real y el coste
    # es despreciable (<100ms por ejecucion incluso con miles de convs).
    cleanup_scheduler = None
    if settings.cleanup_enabled:
        from hermes.scheduler import ConversationCleanupScheduler

        cleanup_scheduler = ConversationCleanupScheduler(
            interval_minutes=settings.cleanup_interval_minutes,
            max_age_minutes=settings.cleanup_max_age_minutes,
            db=db,
        )
        await cleanup_scheduler.start()

    # Sprint 12.1 (TDD_S12_DELETE_AND_SYNC.md §7.4): job diario de
    # hard-delete de tombstoned convs (past retention window).
    # Sin este scheduler, las convs tombstoned se acumulan para siempre
    # (purge_at nunca se aplica). Cubre el caso de la ventana 7d por
    # defecto: tras 7d del soft_delete, la conv se hard-delete via
    # CASCADE en messages.
    tombstone_purge_scheduler = None
    from hermes.scheduler import TombstonePurgeScheduler

    tombstone_purge_scheduler = TombstonePurgeScheduler(
        interval_hours=24,
        db=db,
    )
    await tombstone_purge_scheduler.start()

    # Sprint 17 Slice 1.5 (PR #113b, B2 fix): Vault scheduler para
    # process_inbox + janitor_running_jobs. Sin esto, Slice 1.5
    # GREEN es dead code: los jobs en done/ y processing/ se acumulan
    # sin que nadie los procese. Se instancia IngestRouter UNA vez
    # aquí, con Vault (auto-wired) + FsInboxWriter (real filesystem
    # gateway) + Settings. El mismo router se pasa a Vault para
    # auto-ingest en add() y al VaultScheduler para los jobs.
    vault_scheduler = None
    embed_watcher_scheduler = None
    vault_embedder = None
    ingest_router = None
    vault = None
    try:
        from hermes.memory.ingest_router import FsInboxWriter, IngestRouter
        from hermes.memory.vault import Vault
        from hermes.scheduler import VaultScheduler

        if db is not None:
            vault = Vault(db, ingest_router=None)  # router se inyecta tras crear
            inbox_writer = FsInboxWriter(settings.vault_inbox_root)
            # PR #113b Slice 2.5 (M6 fix): pass db to IngestRouter so
            # _reconcile_db_from_filesystem() can scan ingest_jobs.
            ingest_router = IngestRouter(
                vault=vault,
                inbox=inbox_writer,
                settings=settings,
                db=db,
            )
            # PR #113b (B1): re-bind vault con el router para que
            # add() pueda kick off Tier 0 ingest. Esto es opt-out
            # vía settings.vault_auto_ingest_on_add.
            vault._router = ingest_router
            vault_scheduler = VaultScheduler(
                interval_minutes=5,
                ingest_router=ingest_router,
            )
            await vault_scheduler.start()

            # PR #113c (B2 fix): wire EmbedWatcher + EmbedWatcherScheduler.
            # Sin esto, Slice 2.5 GREEN es dead code: el texto llega
            # via IngestRouter, pero las embeddings nunca se computan
            # y search() retorna [] siempre. VaultEmbedder requiere
            # una EmbeddingsService (construida arriba) y un Chunker.
            # Opt-out: si vault_embedding_model está vacío, no-op.
            if settings.vault_embedding_model:
                from hermes.memory.chunker import Chunker
                from hermes.memory.embedder import EmbedWatcher, VaultEmbedder
                from hermes.scheduler import EmbedWatcherScheduler

                vault_embedder = VaultEmbedder(
                    vault=vault,
                    db=db,
                    embeddings=embeddings_service,
                    chunker=Chunker(
                        max_tokens=settings.vault_chunk_max_tokens,
                        overlap_tokens=settings.vault_chunk_overlap_tokens,
                    ),
                    settings=settings,
                )
                watcher = EmbedWatcher(
                    embedder=vault_embedder,
                    settings=settings,
                    # PR #118 (Sprint 18 hardening, Gemini P0 #2):
                    # pass the GLOBAL stop_event (from
                    # install_signal_handlers) so SIGTERM propagates
                    # to the watcher. Hermes shutdown can now complete
                    # within the Docker grace period (10s default)
                    # instead of waiting 60s+ for the LLM call.
                    stop_event=stop_event,
                )
                embed_watcher_scheduler = EmbedWatcherScheduler(
                    interval_s=settings.vault_watcher_poll_interval_s,
                    watcher=watcher,
                )
                await embed_watcher_scheduler.start()
            else:
                logger.warning(
                    "embed_watcher_scheduler_skipped_no_model",
                    extra={"reason": "vault_embedding_model is empty"},
                )
        else:
            logger.warning("vault_scheduler_skipped_no_db")
    except Exception:
        # PR #113b: si la inicialización del vault/IngestRouter
        # falla (import error, FS permission, etc.), NO tumbar el
        # startup. El resto de Hermes (LLM, HTTP, TG) sigue
        # funcionando. Slice 1.5 queda disabled; operador ve
        # el warning en logs.
        logger.exception("vault_scheduler_init_failed")

    # Sprint 19 Slice 4: drop folder watcher. Detecta archivos nuevos
    # en <vault_drop_root>/<subdir>/, deriva file_id (SHA-256), inserta
    # en vault_files + vault_file_collections, escribe manifest para
    # que process_inbox() (Sprint 17) los procese. La colección se
    # auto-crea si el subdir es nuevo. Idempotente en restart (scan-on-startup).
    drop_watcher_task = None
    edge_coordinator = None
    edge_zombie_scheduler = None
    m6_reconcile_scheduler = None
    ocr_repo = None  # Sprint 19 Slice 4d: used by OCR command router
    if settings.vault_drop_enabled and db is not None:
        try:
            from hermes.memory.collections import VaultCollectionsRepo
            from hermes.memory.drop_watcher import DropWatcher
            from hermes.memory.edge_coordinator import EdgeCoordinator
            from hermes.memory.ocr_pending_repo import OcrPendingRepo
            from hermes.scheduler import EdgeZombieScheduler

            drop_root = settings.vault_drop_root or (settings.vault_inbox_root / "drop")
            collections_repo = VaultCollectionsRepo(db)
            ocr_repo = OcrPendingRepo(db)

            # Sprint 19 Slice 4c: edge coordinator. Only created if
            # EDGE_COMPUTERS is set. Otherwise, the watcher still
            # creates ocr_pending rows but never auto-queues.
            edge_computers = _parse_edge_computers(settings.edge_computers)
            if edge_computers:
                edge_root = drop_root.parent / "_infrastructure"
                edge_coordinator = EdgeCoordinator(
                    edge_computers=edge_computers,
                    db=db,
                    ocr_repo=ocr_repo,
                    edge_root=edge_root,
                    auto_ocr=settings.ocr_auto_edge_ocr,
                    autoqueue_threshold=settings.ocr_edge_autoqueue_threshold,
                    max_queue_size=settings.ocr_edge_max_queue_size,
                    batch_delay_ms=settings.ocr_edge_batch_delay_ms,
                    probe_timeout_s=settings.edge_probe_timeout_s,
                    probe_interval_s=settings.edge_probe_interval_s,
                    smb_root_prefix=settings.edge_smb_root_prefix,
                )
                await edge_coordinator.start()
                logger.info(
                    "edge_coordinator_started",
                    extra={"pcs": [f"{h}:{p}" for h, p in edge_computers]},
                )

                # M6 Phase 5 zombie recovery scheduler
                edge_zombie_scheduler = EdgeZombieScheduler(
                    interval_s=settings.ocr_edge_zombie_scan_interval,
                    coordinator=edge_coordinator,
                    timeout_hours=settings.ocr_edge_timeout_hours,
                )
                await edge_zombie_scheduler.start()
            else:
                logger.info(
                    "edge_coordinator_disabled",
                    extra={"reason": "EDGE_COMPUTERS env var empty"},
                )

            drop_watcher = DropWatcher(
                db=db,
                collections_repo=collections_repo,
                drop_root=drop_root,
                ocr_pending_repo=ocr_repo,
                edge_coordinator=edge_coordinator,
                autoqueue_threshold=settings.ocr_edge_autoqueue_threshold,
                monitor_roots=settings._get_monitor_roots(),
                max_pending=settings.vault_monitor_max_pending,
                # Sprint 19 followup (user feedback 2026-07-12): route
                # root-level files in drop_root to this collection
                # instead of skipping. True opt-in: empty = skip.
                default_collection=settings.vault_drop_default_collection or None,
                # Sprint 19.5 (PR-A): PDF OCR fallback settings
                ocr_fallback_dpi=settings.vault_ocr_fallback_dpi,
                ocr_fallback_grayscale=settings.vault_ocr_fallback_grayscale,
                ocr_fallback_lang=settings.vault_ocr_fallback_lang,
            )
            # Scan on startup (idempotent — handles restart with pending files)
            scan_results = await drop_watcher.scan_existing()
            logger.info(
                "drop_watcher_startup_scan",
                extra={
                    "drop_root": drop_root.as_posix(),
                    "files_processed": len(scan_results),
                },
            )
            # Run the watchfiles loop in a separate task
            drop_watcher_task = asyncio.create_task(
                drop_watcher.run(stop_event=stop_event),
                name="drop_watcher",
            )

            # Sprint 19 Slice 4d v2 commit 5: M6ReconcileScheduler.
            # Re-queues dropped events + detects orphans on a 5-min
            # interval. Works alongside the watcher (which is real-time
            # via watchfiles). M6 is the safety net for queue-full drops.
            from hermes.scheduler import M6ReconcileScheduler

            monitor_roots = settings._get_monitor_roots()
            m6_reconcile_scheduler = M6ReconcileScheduler(
                db=db,
                drop_watcher=drop_watcher,
                monitor_roots=monitor_roots,
                interval_s=300,  # 5 min
            )
            await m6_reconcile_scheduler.start()
        except Exception:
            logger.exception("drop_watcher_init_failed")

    # Sprint 10.4: Push notifications (Telegram health alerts).
    # Requisito previo para Sprint 11 (WebUI SPOF mitigation, ver ADR-004 §5).
    # - TelegramNotifier: envia mensajes via Telegram Bot API.
    #   Lee TELEGRAM_BOT_TOKEN y TELEGRAM_CHAT_ID de env. Si falta alguno,
    #   notifier es no-op (graceful degradation, log warning).
    # - HealthChecker: loop periodico (default 60s) que ejecuta checks
    #   (HTTP API self-ping, disk space, DB integrity) y dispara alerts
    #   via TelegramNotifier cuando falla alguno. asyncio.create_task.
    # - Dedup: 1 alert por alert_type por cooldown window (default 1h)
    #   dentro de TelegramNotifier. Evita spam en outages largos.
    health_checker = None
    health_checker_task: asyncio.Task | None = None

    # The optional application-level egress firewall is disabled by default.
    # Deployments can combine it with their own DNS filtering and host/network
    # firewall policy. The module remains available for explicit opt-in.
    if settings.push_notifications_enabled:
        from hermes.handlers.notifications import TelegramNotifier
        from hermes.health import HealthChecker

        notifier = TelegramNotifier(
            cooldown_seconds=settings.push_notification_cooldown_seconds,
        )
        if not notifier.enabled:
            logger.warning(
                "s10_4_health_alerts_disabled",
                extra={
                    "reason": "TELEGRAM_BOT_TOKEN o TELEGRAM_CHAT_ID no configurados. "
                    "El notifier no enviara alerts. Ver docs/S10.4_SETUP.md (futuro).",
                },
            )
        health_checker = HealthChecker(
            settings=settings,
            notifier=notifier,
            db=db,
            check_interval_seconds=settings.health_check_interval_seconds,
        )
        # asyncio.create_task vs APScheduler: TDD S10.4 v1.3 dice 'APScheduler
        # o asyncio.create_task'. Elegi asyncio.create_task por simplicidad
        # (no requiere manejar lifecycle de scheduler). Trade-off: si el
        # check tarda mas que check_interval, podemos tener checks concurrentes
        # (mitigado: cada check es <100ms, mucho menos que 60s interval).
        health_checker_task = asyncio.create_task(
            health_checker.start(), name="s10_4_health_checker"
        )
        logger.info(
            "s10_4_health_checker_started",
            extra={
                "interval_seconds": settings.health_check_interval_seconds,
                "telegram_enabled": notifier.enabled,
            },
        )

    # Sprint 9.3: Web Search Router (multi-backend, opt-in via search_enabled).
    # Se wire-up DESPUES de llm/embeddings_service porque la inicializacion
    # solo crea objetos in-memory (no hace requests). El tool se registra
    # con tool_registry en el bloque register_builtin_tools de abajo.
    search_budget = None
    search_circuit_breaker = None
    search_concurrency = None
    search_backends: dict = {}
    search_tool_factory = None
    if settings.search_enabled:
        from hermes.services.search.budget import BudgetTracker
        from hermes.services.search.exa import ExaBackend
        from hermes.services.search.resilience import (
            CircuitBreakerRegistry,
            ConcurrencyLimiter,
        )
        from hermes.services.search.searxng import SearXNGBackend
        from hermes.services.search.tavily import TavilyBackend
        from hermes.tools.web_search import make_search_tool_callable

        # 1. BudgetTracker: -1 = ilimitado (SearXNG), 1000/mes para Tavily/Exa.
        search_budget = BudgetTracker(
            db=db,
            limits={
                "searxng": -1,
                "tavily": settings.tavily_monthly_limit,
                "exa": settings.exa_monthly_limit,
            },
        )
        # 2. Circuit breaker + concurrency limiter (compartidos entre backends).
        search_circuit_breaker = CircuitBreakerRegistry(
            threshold=settings.search_circuit_breaker_threshold,
            ttl_seconds=settings.search_circuit_breaker_ttl_seconds,
        )
        # S9.3.1 punto 2: concurrency per-backend con defaults
        # Tavily/Exa son mas permisivos que SearXNG (que depende de engines upstream)
        search_concurrency = ConcurrencyLimiter(
            limits={
                "searxng": settings.search_max_concurrent_searxng,
                "tavily": settings.search_max_concurrent_tavily,
                "exa": settings.search_max_concurrent_exa,
            }
        )
        # 3. Backends opt-in: SearXNG si URL set, Tavily/Exa si API key set.
        if settings.search_searxng_url:
            search_backends["searxng"] = SearXNGBackend(
                url=settings.search_searxng_url,
                budget=search_budget,
                timeout=settings.search_timeout_searxng,
            )
        if settings.tavily_api_key:
            search_backends["tavily"] = TavilyBackend(
                api_key=settings.tavily_api_key,
                budget=search_budget,
                timeout=settings.search_timeout_tavily,
            )
        if settings.exa_api_key:
            search_backends["exa"] = ExaBackend(
                api_key=settings.exa_api_key,
                budget=search_budget,
                timeout=settings.search_timeout_exa,
            )
        # 4. Tool factory: solo si hay al menos un backend (SearXNG default).
        if search_backends:
            search_tool_factory = lambda: make_search_tool_callable(  # noqa: E731
                backends=search_backends,
                budget=search_budget,
                circuit_breaker=search_circuit_breaker,
                semaphore=search_concurrency,
                size_guard_chars=settings.search_size_guard_chars,
                default_num_results=settings.search_default_num_results,
                max_num_results=settings.search_max_num_results,
            )
            logger.info(
                "search_router_initialized",
                extra={"backends": list(search_backends.keys())},
            )

    # Sprint 4 MVP-1: si tools_enabled, construir ToolRegistry con las
    # 4 tools builtin y pasarlo al handler. Si no, el handler usa el
    # path legacy (router.chat directo).
    # Sprint 9.1: register_builtin_tools acepta opcionalmente el
    # EmbeddingsService para registrar search_files si RAG está
    # habilitado.
    tool_registry: ToolRegistry | None = None
    if settings.tools_enabled:
        tool_registry = ToolRegistry()
        register_builtin_tools(
            tool_registry,
            settings=settings,
            db=db,
            router=llm,
            start_time=time.time(),
            embeddings_service=embeddings_service,
            vault_embedder=vault_embedder,
        )
        # Sprint 9.3: registrar hermes_search si search_enabled y hay
        # al menos un backend configurado.
        if search_tool_factory is not None:
            from hermes.tools.web_search import SEARCH_TOOL_SCHEMA

            tool_registry.register(
                "hermes_search",
                search_tool_factory(),
                description=SEARCH_TOOL_SCHEMA["function"]["description"],
                schema=SEARCH_TOOL_SCHEMA["function"]["parameters"],
                tool_category="read",  # web search devuelve contenido externo
            )
            logger.info("search_tool_registered")
        logger.info(
            "tools_registered",
            extra={"tools": tool_registry.list_tools(), "count": len(tool_registry.list_tools())},
        )

    # Sprint 6 T53 v3.1: HTTP API server (opt-in).
    # Si enable_http_api=True, arrancamos uvicorn.Server en una task
    # paralela que comparte los singletons (db, llm, tool_registry).
    # uvicorn.Server (no uvicorn.run) permite shutdown limpio via
    # should_exit = True cuando llega stop_event.
    http_app = None
    http_server = None
    http_task: asyncio.Task | None = None
    if settings.enable_http_api:
        from hermes.receivers.http_api import create_app

        http_app = create_app(
            settings=settings,
            db=db,
            router=llm,
            registry=tool_registry,
            embeddings_service=embeddings_service,
            telemetry=telemetry,  # Sprint 9.3.3: para que AgentLoop registre metricas
            ocr_repo=ocr_repo,  # Sprint 19 Slice 4d: OCR decision API
            edge_coordinator=edge_coordinator,
        )
        import uvicorn

        config = uvicorn.Config(
            http_app,
            host=settings.hermes_api_host,
            port=settings.hermes_api_port,
            loop="asyncio",
            log_level=settings.log_level.lower(),
        )
        http_server = uvicorn.Server(config)
        http_task = asyncio.create_task(http_server.serve(), name="http_api")
        logger.info(
            "http_api_started",
            extra={
                "host": settings.hermes_api_host,
                "port": settings.hermes_api_port,
            },
        )

    # Sprint 11 (ADR-004): WebUI-primary. Gate del PollingReceiver.
    # Solo arranca si enable_telegram=True Y telegram_bot_token está
    # Sprint 19 Slice 4d (B1 fix): instantiate the OcrProvider. The
    # provider-agnostic HostedLlmOcrProvider wraps the existing LLM
    # client (provider-agnostic via the same chain). The actual model
    # is whatever Settings.llm_text_primary resolves to (default
    # MiniMax-M3, but no hardcoded dependency). Sprint 22+ can swap to
    # LocalVisionOcrProvider (LLaVA via Ollama) without changing the
    # decision logic.
    ocr_provider = None
    if settings.ocr_default_provider == "hosted_llm" and llm is not None:
        from hermes.llm.ocr import HostedLlmOcrProvider

        ocr_provider = HostedLlmOcrProvider(llm)
        logger.info(
            "ocr_provider_initialized",
            extra={"provider": ocr_provider.name, "model_chain": settings.text_chain},
        )
    else:
        logger.warning(
            "ocr_provider_not_initialized",
            extra={
                "configured": settings.ocr_default_provider,
                "llm_available": llm is not None,
                "reason": "provider-agnostic: Sprint 22+ will add local/edge "
                "providers. /externalOCR will return a config error until then.",
            },
        )

    # configurado. Si enable_telegram=True pero el token falta, warning
    # y continuamos sin Telegram (HTTP API + HealthChecker siguen
    # funcionando, el bot no se conecta). Esto permite rollouts
    # graduales y environments mixtos (e.g. CI sin token).
    receiver = None
    receiver_task: asyncio.Task | None = None
    if settings.enable_telegram:
        if settings.telegram_bot_token:
            receiver = PollingReceiver(
                bot_token=settings.telegram_bot_token,
                allowed_user_ids=settings.allowed_user_ids_list,
                db=db,
                settings=settings,
                telemetry=telemetry,
                tool_registry=tool_registry,
                embeddings_service=embeddings_service,  # Sprint 16 (US-3.2)
                # Sprint 19 Slice 4d: OCR user commands. ocr_repo + edge_coordinator
                # are created above (in the vault_drop_enabled block). If
                # drop is disabled, they are None and the OCR router is
                # not registered (commands return usage help).
                ocr_repo=ocr_repo,
                edge_coordinator=edge_coordinator,
                # B1 fix: provider-agnostic OcrProvider (required for
                # /externalOCR 2-step flow). None if no LLM available.
                ocr_provider=ocr_provider,
            )
            logger.info(
                "telegram_receiver_started",
                extra={
                    "allowed_user_ids_count": len(settings.allowed_user_ids_list),
                },
            )
        else:
            logger.warning(
                "s11_telegram_receiver_disabled",
                extra={
                    "reason": "enable_telegram=True pero TELEGRAM_BOT_TOKEN no "
                    "configurado. Bot no conectara. HTTP API y HealthChecker "
                    "siguen funcionando. Para activar Telegram, definir "
                    "TELEGRAM_BOT_TOKEN en .env (y ALLOWED_USER_IDS).",
                },
            )
    else:
        logger.info(
            "s11_telegram_receiver_skipped",
            extra={
                "reason": "enable_telegram=False. WebUI es la interfaz primary "
                "(Sprint 11 ADR-004). Telegram es opt-in legacy.",
            },
        )

    # PR #118: stop_event is now created EARLY (top of run()) so the
    # EmbedWatcher wiring below can observe it. See the comment near
    # the top of run() for context.
    if receiver is not None:
        receiver_task = asyncio.create_task(receiver.run_forever(stop_event), name="polling")

    try:
        await stop_event.wait()
    finally:
        logger.info("hermes_stopping")
        # Sprint 6 T53 v3.1: parar uvicorn ANTES de cerrar DB/llm
        # para que requests en vuelo puedan terminar limpiamente.
        # uvicorn.Server.serve() respeta should_exit = True y para.
        if http_server is not None:
            http_server.should_exit = True
        if http_task is not None:
            with suppress(asyncio.CancelledError, TimeoutError):
                await http_task
        if receiver_task is not None:
            receiver_task.cancel()
            with suppress(asyncio.CancelledError, TimeoutError):
                await receiver_task
        if health is not None:
            await health.stop()
    # Sprint 8 S8.4: parar backup scheduler (espera al job en curso).
    # Sprint 9.2: parar sleep cycle scheduler si esta activo.
    # Sprint 9.4: parar conversation cleanup scheduler.
    # Sprint 10.4: parar health checker (signal stop, esperar task).
    # Orden: schedulers -> telemetry -> llm -> db.
    if backup_scheduler is not None:
        await backup_scheduler.shutdown()
    if sleep_cycle_scheduler is not None:
        await sleep_cycle_scheduler.shutdown()
    if cleanup_scheduler is not None:
        await cleanup_scheduler.shutdown()
    if tombstone_purge_scheduler is not None:
        await tombstone_purge_scheduler.shutdown()
    if vault_scheduler is not None:
        await vault_scheduler.shutdown()
    # PR #113c (B2 fix): shutdown the embed watcher scheduler
    # AFTER vault_scheduler (it depends on vault being alive).
    # PR #118 (Sprint 18 hardening, Gemini P0 #2): pass timeout_s=10
    # so the scheduler's watcher.shutdown() can drain in-flight embed
    # within the Docker SIGTERM grace period. Without this, APScheduler's
    # default wait=True blocks until the LLM call completes (60+ seconds
    # on OpenRouter free tier) → Docker SIGKILL → DB poisoning risk.
    if embed_watcher_scheduler is not None:
        await embed_watcher_scheduler.shutdown(timeout_s=10.0)
    # Sprint 19 Slice 4: cancel drop watcher task (the stop_event is
    # already set by install_signal_handlers, so run() exits cleanly).
    if drop_watcher_task is not None and not drop_watcher_task.done():
        with suppress(asyncio.CancelledError, TimeoutError):
            await asyncio.wait_for(drop_watcher_task, timeout=5.0)
        logger.info("drop_watcher_stopped")
    # Sprint 19 Slice 4c: stop edge coordinator + zombie scheduler.
    if edge_zombie_scheduler is not None:
        await edge_zombie_scheduler.shutdown()
    # Sprint 19 Slice 4d v2: stop M6 reconciliation scheduler
    if m6_reconcile_scheduler is not None:
        await m6_reconcile_scheduler.shutdown()
    if edge_coordinator is not None:
        await edge_coordinator.stop()
    if health_checker is not None:
        health_checker.stop()
        if health_checker_task is not None and not health_checker_task.done():
            with suppress(asyncio.CancelledError, TimeoutError):
                await asyncio.wait_for(health_checker_task, timeout=5.0)
        logger.info("s10_4_health_checker_stopped")
    await _close_core_resources(
        embeddings_service=embeddings_service, telemetry=telemetry, llm=llm, db=db
    )
    logger.info("hermes_stopped")

    return 0


def main() -> int:
    try:
        return asyncio.run(run())
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
