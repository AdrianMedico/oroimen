"""Phase 4: P1 failure modes + multi-tier orchestration (TDD §5 P1).

Validates that the multi-tier embedding router:

**Failure modes** (graceful degradation):
- NAS down -> router still works (no chat RAG; degraded)
- Edge down -> vault_ingest retries with cloud
- Cloud down -> vault_ingest fails gracefully (returns error or cached)
- All down -> router reports not enabled, embed() raises AllTiersFailed

**Multi-tier orchestration** (Rule 3: within-policy dim match):
- chat_rag=nas -> 768-dim query
- vault_ingest=cloud,edge -> uses cloud first, edge as fallback
- chat_rag=cloud,cloud (single tier) -> works
- chat_rag=nas,edge (dim mismatch: 768 vs 4096) -> hard-fail at startup

These tests use the failable backend stubs (added to conftest.py in
Phase 4) to simulate tier outages.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from hermes.services.embedding_router import (
    AllTiersFailed,
    EmbeddingBackend,
    EmbeddingPolicy,
    EmbeddingRouter,
    PolicyConfig,
)

pytestmark = pytest.mark.e2e


# ---------------------------------------------------------------------------
# Multi-tier orchestration
# ---------------------------------------------------------------------------


def test_chat_rag_single_tier_returns_correct_dim(
    embedding_router_factory: object,
) -> None:
    """P1 orchestration: chat_rag=nas returns 768-dim vector.

    Validates the most common production case: chat RAG with only
    NAS tier enabled (Sprint 19.5 default).
    """
    router = embedding_router_factory()
    result = asyncio.run(router.embed("hola mundo"))
    assert result.shape == (768,), f"Expected 768-dim, got {result.shape}"
    assert result.dtype == np.float32


def test_router_reports_enabled_when_at_least_one_tier_up(
    embedding_router_factory: object,
) -> None:
    """P1: router.is_enabled is True if any tier is up (canonical case)."""
    router = embedding_router_factory()
    assert router.is_enabled, "Router with all canonical tiers should report enabled"


def test_router_reports_disabled_when_all_tiers_down(
    failable_nas: object,
    failable_edge: object,
    failable_cloud: object,
) -> None:
    """P1: router.is_enabled is False when all tiers are unhealthy.

    Used by callers like AgentLoop.run() to decide whether to even
    attempt RAG. If disabled, retrieval is skipped and chat works
    without RAG (graceful degradation).
    """
    failable_nas.healthy = False
    failable_edge.healthy = False
    failable_cloud.healthy = False

    backends: dict[str, EmbeddingBackend] = {
        "nas": failable_nas,
        "edge": failable_edge,
        "cloud": failable_cloud,
    }
    policies: dict[EmbeddingPolicy, PolicyConfig] = {
        EmbeddingPolicy.CHAT_RAG: PolicyConfig(use_case=EmbeddingPolicy.CHAT_RAG, tiers=["nas"]),
        EmbeddingPolicy.VAULT_INGEST: PolicyConfig(
            use_case=EmbeddingPolicy.VAULT_INGEST, tiers=["cloud", "edge"]
        ),
    }
    router = EmbeddingRouter(backends=backends, policies=policies)
    assert not router.is_enabled, "Router with all tiers down should report disabled"


# ---------------------------------------------------------------------------
# Failure modes: cascade
# ---------------------------------------------------------------------------


def test_nas_down_chat_rag_returns_error(
    failable_nas: object,
    failable_edge: object,
    failable_cloud: object,
) -> None:
    """P1 failure mode: chat_rag=nas, NAS down -> embed raises.

    chat_rag policy has only NAS tier. If NAS is down, there's no
    fallback (the canonical chat_rag is single-tier for low latency).
    The router raises AllTiersFailed; the caller (AgentLoop) catches
    it and continues without RAG.
    """
    failable_nas.healthy = False

    backends: dict[str, EmbeddingBackend] = {
        "nas": failable_nas,
        "edge": failable_edge,
        "cloud": failable_cloud,
    }
    policies: dict[EmbeddingPolicy, PolicyConfig] = {
        EmbeddingPolicy.CHAT_RAG: PolicyConfig(use_case=EmbeddingPolicy.CHAT_RAG, tiers=["nas"]),
    }
    router = EmbeddingRouter(backends=backends, policies=policies)

    with pytest.raises(AllTiersFailed):
        asyncio.run(router.embed("test query"))


def test_vault_ingest_falls_back_from_cloud_to_edge(
    failable_nas: object,
    failable_edge: object,
    failable_cloud: object,
) -> None:
    """P1 failure mode: vault_ingest=cloud,edge. Cloud raises -> falls to edge.

    Validates the cascade: the router tries cloud first, gets a
    RuntimeError, then tries edge. Edge succeeds and returns the
    4096-dim vector.

    Note: cloud is ``healthy=True`` but ``raise_on_call=True`` --
    reports as enabled (so the router TRIES it) but the call fails.
    This is the cascade path (vs the disabled-skip path).
    """
    failable_cloud.healthy = True
    failable_cloud.raise_on_call = True
    failable_edge.healthy = True
    failable_edge.raise_on_call = False

    backends: dict[str, EmbeddingBackend] = {
        "nas": failable_nas,
        "edge": failable_edge,
        "cloud": failable_cloud,
    }
    policies: dict[EmbeddingPolicy, PolicyConfig] = {
        EmbeddingPolicy.VAULT_INGEST: PolicyConfig(
            use_case=EmbeddingPolicy.VAULT_INGEST, tiers=["cloud", "edge"]
        ),
    }
    router = EmbeddingRouter(backends=backends, policies=policies)

    result = asyncio.run(router.embed("ingest text", EmbeddingPolicy.VAULT_INGEST))
    assert result.shape == (4096,), f"Expected 4096-dim from edge fallback, got {result.shape}"
    assert failable_cloud.call_count == 1
    assert failable_edge.call_count == 1


def test_vault_ingest_all_tiers_down_raises(
    failable_nas: object,
    failable_edge: object,
    failable_cloud: object,
) -> None:
    """P1 failure mode: vault_ingest=cloud,edge, all raise -> AllTiersFailed."""
    failable_edge.raise_on_call = True
    failable_cloud.raise_on_call = True

    backends: dict[str, EmbeddingBackend] = {
        "nas": failable_nas,
        "edge": failable_edge,
        "cloud": failable_cloud,
    }
    policies: dict[EmbeddingPolicy, PolicyConfig] = {
        EmbeddingPolicy.VAULT_INGEST: PolicyConfig(
            use_case=EmbeddingPolicy.VAULT_INGEST, tiers=["cloud", "edge"]
        ),
    }
    router = EmbeddingRouter(backends=backends, policies=policies)

    with pytest.raises(AllTiersFailed):
        asyncio.run(router.embed("ingest text", EmbeddingPolicy.VAULT_INGEST))


# ---------------------------------------------------------------------------
# Smoke: cache layer (P3 preview)
# ---------------------------------------------------------------------------


def test_same_text_produces_deterministic_embedding(
    embedding_router_factory: object,
) -> None:
    """P3 preview: same input text -> same output vector (cacheable).

    The deterministic stub proves embeddings are reproducible. A real
    cache layer (Phase 5) can use this property to short-circuit
    identical queries.
    """
    router = embedding_router_factory()
    v1 = asyncio.run(router.embed("identical text"))
    v2 = asyncio.run(router.embed("identical text"))
    np.testing.assert_array_equal(v1, v2)


def test_different_text_produces_different_embedding(
    embedding_router_factory: object,
) -> None:
    """P3 preview: different input text -> different output vector.

    Sanity check that the deterministic stub is actually text-dependent
    (not just returning a constant).
    """
    router = embedding_router_factory()
    v1 = asyncio.run(router.embed("text A"))
    v2 = asyncio.run(router.embed("text B"))
    assert not np.allclose(v1, v2), "Different texts should produce different embeddings"
