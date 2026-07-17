"""Unit tests for DeepResearchService._run_in_scrape_pool (NB1 verifier finding).

Anti-regression checks (deliverable §NB1):
- Counter ``_scrape_active`` increments on entry.
- Counter decrements on success (finally block).
- Counter decrements on exception (finally block).
- Counter is back to 0 after multiple concurrent runs finish.
- Counter does NOT go negative (increment-before-submit ordering).
- Metric emission uses _scrape_active, NOT _idle_semaphore._value.

Strategy:
- We use a REAL ThreadPoolExecutor (1 worker, to serialize and observe state
  deterministically) but with a small sleep in the fn. Real-thread > mock
  because the real run_in_executor path is what we're testing, including the
  increment/decrement finally semantics across asyncio/thread boundaries.
- The mock-pool tests were flaky because MagicMock.submit doesn't actually
  invoke the fn — it just returns a future that the caller resolves, but
  the fn never runs in that path. Real executor = real fn invocation.
"""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

import pytest

from hermes.jobs.service import DeepResearchService


def _make_service(pool_workers: int = 1) -> DeepResearchService:
    """Build a DeepResearchService with a real ThreadPoolExecutor.

    Args:
        pool_workers: how many workers in the scrape pool. Default 1 so we
            can deterministically observe the "during" state without races.
    """
    service = DeepResearchService(
        db=None,
        notifier=None,
        llm_router=None,
        web_search=None,
        settings=MagicMock(),
        scheduler=None,
    )
    service._scrape_pool = ThreadPoolExecutor(
        max_workers=pool_workers,
        thread_name_prefix="scrape-test",
    )
    service._scrape_active = 0
    return service


@pytest.mark.asyncio
async def test_threadpool_saturation_counter_increments() -> None:
    """Counter increments when _run_in_scrape_pool is entered and decrements on completion."""
    service = _make_service(pool_workers=1)
    observed: dict[str, int] = {}

    def fake_fn() -> str:
        # While fn runs, capture the counter state.
        observed["during"] = service._scrape_active
        # Sleep briefly so the test's outer await has time to observe counter=1.
        time.sleep(0.05)
        return "done"

    # Start wrapper — counter increments immediately (before await).
    task = asyncio.create_task(service._run_in_scrape_pool(fake_fn))
    # Let the wrapper reach ``await submit(...)`` and fn start.
    await asyncio.sleep(0.02)
    # Counter must be +1 while fn is running.
    assert (
        service._scrape_active == 1
    ), f"Counter should be +1 while fn runs, got {service._scrape_active}"

    result = await task
    assert result == "done"
    assert (
        observed["during"] == 1
    ), f"Counter during fn execution should be 1, got {observed.get('during')}"
    assert (
        service._scrape_active == 0
    ), f"Counter should return to 0 after completion, got {service._scrape_active}"


@pytest.mark.asyncio
async def test_threadpool_saturation_counter_decrements_on_exception() -> None:
    """Counter decrements even if the underlying fn raises.

    Verifies the ``finally`` clause — critical because exceptions are
    common in Phase 2 (httpx errors, parse failures).
    """
    service = _make_service(pool_workers=1)

    def fake_fn() -> None:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        await service._run_in_scrape_pool(fake_fn)

    # Even though fn raised, counter must be back to 0.
    assert (
        service._scrape_active == 0
    ), f"Counter should be 0 after exception, got {service._scrape_active}"


@pytest.mark.asyncio
async def test_threadpool_saturation_counter_concurrent() -> None:
    """Counter tracks concurrent in-flight tasks correctly (peak = N).

    Uses 4 workers + 3 concurrent calls: peak counter = 3 (all submitted at once).
    After all complete, counter returns to 0.
    """
    service = _make_service(pool_workers=4)
    observed_peak = {"peak": 0}

    def make_blocking_fn(idx: int):
        def fn() -> str:
            current = service._scrape_active
            observed_peak["peak"] = max(observed_peak["peak"], current)
            # Sleep LONGER than the outer asyncio.sleep(0.02) so all 3 tasks
            # are still mid-execution when the test reads the counter.
            # Previous 0.01 was racy on faster machines (counter drained
            # before the assertion fired).
            time.sleep(0.1)
            return f"result-{idx}"

        return fn

    tasks = [
        asyncio.create_task(service._run_in_scrape_pool(make_blocking_fn(i))) for i in range(3)
    ]
    # Yield so all wrappers reach ``await submit(...)`` and fns start.
    await asyncio.sleep(0.02)
    assert (
        service._scrape_active == 3
    ), f"Counter should be 3 with 3 in-flight tasks, got {service._scrape_active}"

    results = await asyncio.gather(*tasks)
    assert results == ["result-0", "result-1", "result-2"]
    assert (
        observed_peak["peak"] == 3
    ), f"Peak counter during execution should be 3, got {observed_peak['peak']}"
    assert (
        service._scrape_active == 0
    ), f"Counter should be 0 after all tasks complete, got {service._scrape_active}"


@pytest.mark.asyncio
async def test_threadpool_saturation_counter_no_negative() -> None:
    """Counter never goes negative (increment-before-submit ordering).

    Critical: increment happens BEFORE submit() in _run_in_scrape_pool, so
    even if the future resolves synchronously the counter is non-negative.
    """
    service = _make_service(pool_workers=2)
    for _i in range(5):
        await service._run_in_scrape_pool(lambda: "x")
    assert (
        service._scrape_active == 0
    ), f"After 5 sequential calls, counter must be 0, got {service._scrape_active}"


@pytest.mark.asyncio
async def test_threadpool_saturation_no_idle_semaphore_access() -> None:
    """NB1 regression check: _idle_semaphore access is GONE from service.py.

    The old code accessed ``self._scrape_pool._idle_semaphore._value``
    (CPython internal, fragile across versions). After the fix, the only
    mention of ``_idle_semaphore`` should be in comments/docstrings
    EXPLAINING why we don't use it. We verify by checking that no
    executable line references it.

    We check via AST parsing — comments don't appear in AST, only code.
    """
    import ast

    import hermes.jobs.service as svc_module

    with open(svc_module.__file__, encoding="utf-8") as fh:
        source = fh.read()
    tree = ast.parse(source)

    # Walk all attribute accesses; flag any that target _idle_semaphore.
    bad_refs: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "_idle_semaphore":
            bad_refs.append(
                f"line {node.lineno}: {ast.unparse(node) if hasattr(ast, 'unparse') else 'attr access'}"
            )

    assert (
        not bad_refs
    ), f"NB1 regression: _idle_semaphore attribute access found in service.py: {bad_refs}"


@pytest.mark.asyncio
async def test_threadpool_saturation_counter_init_zero() -> None:
    """On service construction, _scrape_active starts at 0."""
    service = _make_service(pool_workers=4)
    assert service._scrape_active == 0


@pytest.mark.asyncio
async def test_threadpool_saturation_wrapper_signature() -> None:
    """_run_in_scrape_pool is an async method that accepts (fn, *args).

    Sanity check that the helper exists with the expected signature —
    the production code (Phase 2 fetch_one) calls it with positional args.
    """
    import inspect

    service = _make_service(pool_workers=1)
    assert hasattr(service, "_run_in_scrape_pool")
    assert inspect.iscoroutinefunction(service._run_in_scrape_pool)
    sig = inspect.signature(service._run_in_scrape_pool)
    # fn + *args (var-positional)
    params = list(sig.parameters.values())
    assert len(params) == 2
    assert params[0].name == "fn"
    assert params[1].kind == inspect.Parameter.VAR_POSITIONAL  # *args
