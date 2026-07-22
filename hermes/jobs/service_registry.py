"""Process-local registry for the live ``DeepResearchService``.

The registry is intentionally narrow: it only holds the single
``DeepResearchService`` instance the daemon is currently running
with. The dispatcher (``hermes.jobs.dispatcher``) reads from
this registry at call time so that the APScheduler jobstore
never has to pickle the service graph.

Lifecycle
---------

The registry is a process-local mutable slot. The canonical
write site is ``hermes/__main__.py:startup()`` after the runtime
is fully composed and BEFORE the scheduler is started. The
canonical clear site is during the same lifecycle's
``_deep_research_cleanup`` and on daemon shutdown.

The registry is single-slot, not a list. A re-set replaces the
previous value. This is the intentional design: the
``DeepResearchService`` is a singleton; the registry is the
slot the dispatcher reads from.

Why process-local
=================

- A cross-process registry (file, Redis, sqlite) would
  introduce a new failure domain and would not be picklable
  for free. APScheduler's jobstore is per-process; a per-process
  registry matches the scope of the failure.
- A global module-level mutable is the simplest possible
  implementation. We use a module-private name and expose
  only the typed setters/getters/clearers.
- The slot is typed (``DeepResearchService | None``) so a
  caller that holds a stale reference gets a clear
  ``DispatcherRegistryMissing`` from the dispatcher rather than
  an opaque ``AttributeError``.

Why not a global service locator
================================

A general service locator (e.g. an injected ``ServiceContainer``
holding LLM, search, settings, notifier, fetcher) would invite
future code paths to register arbitrary live objects and
eventually re-serialize them into the APScheduler jobstore. The
narrow typed slot here only accepts the
``DeepResearchService``. The LLM, search, settings, notifier
and fetcher all live on the service instance and never need
their own registry.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hermes.jobs.service import DeepResearchService

logger = logging.getLogger(__name__)

_research_service: DeepResearchService | None = None


def set_research_service(service: DeepResearchService) -> None:
    """Register the live ``DeepResearchService``.

    Called from ``hermes/__main__.py:startup()`` after the runtime
    is fully composed and BEFORE
    ``DeepResearchScheduler.start()``. Calling this twice
    replaces the previous reference (the previous reference is
    silently discarded; the orchestrator is responsible for
    not double-registering).
    """
    global _research_service
    _research_service = service
    logger.info("research_service_registered")


def get_research_service() -> DeepResearchService | None:
    """Return the live ``DeepResearchService`` or ``None``.

    Called by ``hermes.jobs.dispatcher.execute_research_job`` at
    scheduler firing time. ``None`` means the service was never
    registered in this process or was cleared during shutdown.
    """
    return _research_service


def clear_research_service() -> None:
    """Clear the registry. Idempotent.

    Called from ``_deep_research_cleanup`` and from the daemon
    shutdown seam. A clear MUST happen during rollback so a
    subsequent dispatcher run cannot resolve a stale service
    instance whose database connection has already been closed.
    """
    global _research_service
    _research_service = None
    logger.info("research_service_cleared")
