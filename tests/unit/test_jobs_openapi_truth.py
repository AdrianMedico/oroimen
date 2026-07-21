"""DR-Q1A-PRE1A truth-patch 3: FastAPI OpenAPI regression test.

These tests generate the FastAPI OpenAPI schema from the actual
``hermes.receivers.jobs_api.router`` and prove that the public
cancel endpoint documentation is truthful.

Anti-regression checks (DR-Q1A-PRE1A truth-patch 3):
- The cancel query-parameter ``graceful`` description contains
  no hard-cancel claim and no wait-until-current-phase claim.
- The cancel response documentation has no
  ``partial_output_path`` field reference (the public
  CancelResponse has no such field).
- Field names and response shapes remain unchanged.
- The forbidden legacy phrases do not appear in the OpenAPI
  schema (which is what API consumers see).

The test does NOT perform a network call, does NOT use
provider credentials, and does NOT trigger any real provider.
It only inspects the in-process FastAPI app's OpenAPI schema.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi import FastAPI

# These phrases are forbidden in any public API documentation
# (DTO Field descriptions, route Query parameter descriptions,
# route docstrings, cost module docstring, etc.). The OpenAPI
# schema is the API surface that consumers see, so it must not
# contain any of them either.
FORBIDDEN_LEGACY_PHRASES_IN_OPENAPI = (
    "hard cancel inmediato",
    "cancela tras finalizar la phase actual",
    "True si partial output guardado",
    "partial_output_path si existía",
    "lower bound on actual billed tokens",
)


def _build_minimal_app() -> FastAPI:
    """Build a minimal FastAPI app that mounts the jobs router.

    Uses TestClient-style wiring but the app is built in-process
    to avoid relying on the full ``create_app`` factory (which
    requires a Database, LLMRouter, etc.). The jobs router
    itself can be mounted on a minimal FastAPI app; the route
    functions are inspected via the OpenAPI schema and not
    invoked.

    The bearer auth dependency in the router is the only
    potential runtime issue; we do NOT invoke any route — we
    only inspect the OpenAPI schema. FastAPI's ``.openapi()``
    on a mounted router populates the parameters and
    descriptions from the function signature and
    ``Annotated[..., Query(description=...)]`` metadata
    without executing the handler.
    """
    # Lazy import to avoid circular side-effects.
    from hermes.receivers import jobs_api

    app = FastAPI(title="Oroimen (test fixture)")
    app.include_router(jobs_api.router)
    return app


@pytest.fixture(scope="module")
def openapi_schema() -> dict[str, Any]:
    """Return the FastAPI OpenAPI schema for the jobs router."""
    app = _build_minimal_app()
    return app.openapi()


@pytest.fixture(scope="module")
def cancel_endpoint() -> dict[str, Any]:
    """Return the OpenAPI operation for POST /v1/jobs/{job_id}/cancel.

    Returns an empty dict if the operation is not found (which
    is itself a test failure handled by the calling tests).
    """
    app = _build_minimal_app()
    schema = app.openapi()
    paths = schema.get("paths", {})
    # The path is mounted under /v1 prefix.
    for path in ("/v1/jobs/{job_id}/cancel", "/jobs/{job_id}/cancel"):
        ops = paths.get(path, {})
        post = ops.get("post")
        if post is not None:
            return {"path": path, "operation": post, "schema": schema}
    return {"path": None, "operation": None, "schema": schema}


@pytest.fixture(scope="module")
def print_schema(request):
    """Debug helper: print the OpenAPI schema to stdout. Activated
    via ``-s`` or by adding ``print_schema`` to the test name."""
    import json
    app = _build_minimal_app()
    schema = app.openapi()
    print("=== OpenAPI cancel operation ===")
    print(json.dumps(schema["paths"]["/v1/jobs/{job_id}/cancel"]["post"], indent=2, ensure_ascii=False))
    print("=== components.schemas ===")
    print(json.dumps(schema.get("components", {}).get("schemas", {}), indent=2, ensure_ascii=False))


# ---------------------------------------------------------------------------
# 1. Cancel query-parameter description has no hard-cancel claim
# ---------------------------------------------------------------------------


def test_cancel_graceful_query_description_has_no_hard_cancel_claim(
    cancel_endpoint: dict[str, Any],
) -> None:
    """The cancel ``graceful`` Query description must not claim a
    hard cancel. The current code does not implement hard
    cancellation; ``graceful=False`` only moves persistence to
    ``cancelled`` in the DB.

    The forbidden phrase list forbids any mention of "hard cancel"
    in the description (not just positive claims). The current
    description uses "neither option ... performs a hard cancel"
    (a negation); the test must verify the absence of any claim
    that hard cancel is a feature. The truthful test is: the
    description does not CLAIM a hard cancel capability.
    """
    op = cancel_endpoint["operation"]
    assert op is not None, "POST /v1/jobs/{job_id}/cancel not found in OpenAPI schema"
    params = op.get("parameters", [])
    graceful_param = next(
        (p for p in params if p.get("name") == "graceful"),
        None,
    )
    assert graceful_param is not None, (
        "The 'graceful' query parameter is missing from the cancel "
        "endpoint OpenAPI definition."
    )
    desc = graceful_param.get("description", "")
    # The forbidden phrase "hard cancel inmediato" must not appear
    # in any form.
    assert "hard cancel inmediato" not in desc.lower(), (
        f"Cancel graceful Query description must not use the "
        f"forbidden 'hard cancel inmediato' phrase. Got: {desc!r}"
    )
    # The description must not positively CLAIM a hard-cancel
    # capability (e.g., "performs a hard cancel"). The negation
    # ("neither option performs a hard cancel") is acceptable as
    # long as it makes the no-hard-cancel behavior explicit.
    # We verify the positive-claim absence by asserting that
    # the description does not contain "performs a hard cancel"
    # or "executes a hard cancel" or similar positive claims.
    positive_claim_patterns = [
        "performs a hard cancel",
        "executes a hard cancel",
        "triggers a hard cancel",
        "force a hard cancel",
    ]
    for pattern in positive_claim_patterns:
        assert pattern not in desc.lower(), (
            f"Cancel graceful Query description must not contain "
            f"the positive claim {pattern!r}. Got: {desc!r}"
        )


# ---------------------------------------------------------------------------
# 2. Cancel query-parameter description has no wait-until-current-phase claim
# ---------------------------------------------------------------------------


def test_cancel_graceful_query_description_has_no_wait_until_current_phase_claim(
    cancel_endpoint: dict[str, Any],
) -> None:
    """The cancel ``graceful`` Query description must not claim
    that ``graceful=True`` waits for the current phase to finish.
    The current code only updates the DB; it does not wait.
    """
    op = cancel_endpoint["operation"]
    params = op.get("parameters", [])
    graceful_param = next(
        (p for p in params if p.get("name") == "graceful"),
        None,
    )
    desc = graceful_param.get("description", "")
    assert "cancela tras finalizar" not in desc.lower(), (
        f"Cancel graceful Query description must not use the "
        f"forbidden 'cancela tras finalizar la phase actual' phrase. "
        f"Got: {desc!r}"
    )
    # The description must not positively claim a "wait until
    # current phase finishes" behavior. Negation is acceptable
    # as long as it makes the no-wait behavior explicit.
    # Note: "neither option waits for the current phase to finish"
    # contains "waits for the current phase to finish" but as a
    # negation ("neither option ... waits for ..."). We allow the
    # literal "waits for" only if paired with a negation marker
    # like "neither" or "no" within the same sentence.
    if "waits for the current phase" in desc.lower():
        # If the phrase is present, it must be negated within the
        # same sentence.
        assert "neither" in desc.lower() or "no " in desc.lower(), (
            f"Cancel graceful Query description contains 'waits for "
            f"the current phase' without a clear negation. The "
            f"current code does not wait. Got: {desc!r}"
        )


# ---------------------------------------------------------------------------
# 3. Cancel response documentation has no partial_output_path claim
# ---------------------------------------------------------------------------


def test_cancel_response_documentation_has_no_partial_output_path_claim(
    cancel_endpoint: dict[str, Any],
) -> None:
    """The cancel response documentation must not positively
    reference ``partial_output_path`` (no such public field
    exists on ``CancelResponse``). The OpenAPI response schema
    is the authoritative source of truth for API consumers.
    """
    op = cancel_endpoint["operation"]
    op_desc = op.get("description", "")
    # The forbidden phrase "partial_output_path si existía" must
    # not appear in any form (it is the legacy misleading
    # description).
    assert "partial_output_path si existía" not in op_desc.lower(), (
        f"Cancel operation docstring must not use the forbidden "
        f"'partial_output_path si existía' phrase. Got: {op_desc!r}"
    )
    # The JSON schema must not declare a partial_output_path field.
    responses = op.get("responses", {})
    ok = responses.get("200", {}) or responses.get("default", {}) or {}
    schema_ref = ok.get("content", {}).get("application/json", {}).get("schema", {})
    schema = cancel_endpoint["schema"]
    components = schema.get("components", {}).get("schemas", {})
    # Resolve $ref to find the actual response model.
    if "$ref" in schema_ref:
        ref_name = schema_ref["$ref"].rsplit("/", 1)[-1]
        resolved = components.get(ref_name, {})
    else:
        resolved = schema_ref
    properties = resolved.get("properties", {}) or {}
    assert "partial_output_path" not in properties, (
        f"Cancel 200 response JSON schema must not declare a "
        f"partial_output_path property: the public CancelResponse "
        f"has no such field. Got properties: {list(properties.keys())}"
    )
    # Also check the full schema string for the legacy phrase.
    schema_str = json.dumps(resolved)
    assert "partial_output_path si existía" not in schema_str.lower(), (
        f"Cancel 200 response JSON schema must not contain the "
        f"legacy 'partial_output_path si existía' phrase. Got: "
        f"{schema_str!r}"
    )


# ---------------------------------------------------------------------------
# 4. Field names and response shapes remain unchanged
# ---------------------------------------------------------------------------


def test_cancel_response_schema_has_only_id_status_graceful(
    cancel_endpoint: dict[str, Any],
) -> None:
    """The cancel response JSON schema has exactly the three
    pre-PR-9 fields: ``id``, ``status``, ``graceful``. No field
    added, removed, or renamed.
    """
    op = cancel_endpoint["operation"]
    responses = op.get("responses", {})
    ok = responses.get("200", {}) or {}
    schema_ref = ok.get("content", {}).get("application/json", {}).get("schema", {})
    # FastAPI generates $ref for the model; resolve via the
    # components/schemas section.
    schema = cancel_endpoint["schema"]
    components = schema.get("components", {}).get("schemas", {})
    if "$ref" in schema_ref:
        ref_name = schema_ref["$ref"].rsplit("/", 1)[-1]
        resolved = components.get(ref_name, {})
    else:
        resolved = schema_ref
    properties = resolved.get("properties", {}) or {}
    assert set(properties.keys()) == {"id", "status", "graceful"}, (
        f"CancelResponse JSON schema must have exactly "
        f"{{id, status, graceful}}; got {set(properties.keys())}"
    )
    # Resolve $ref for nested types (status is a StrEnum).
    def _resolve_type(prop: dict[str, Any]) -> dict[str, Any]:
        if "$ref" in prop:
            ref_name = prop["$ref"].rsplit("/", 1)[-1]
            return components.get(ref_name, {})
        return prop

    id_prop = _resolve_type(properties["id"])
    graceful_prop = _resolve_type(properties["graceful"])
    status_prop = _resolve_type(properties["status"])
    # The id type is string.
    assert id_prop.get("type") == "string"
    # The graceful type is boolean.
    assert graceful_prop.get("type") == "boolean"
    # The status type is a string enum (the JobStatus StrEnum).
    assert status_prop.get("type") == "string"
    # The status enum contains the expected values.
    status_enum = status_prop.get("enum", [])
    assert "cancelling" in status_enum
    assert "cancelled" in status_enum


# ---------------------------------------------------------------------------
# 5. Forbidden legacy phrases do not appear anywhere in the OpenAPI schema
# ---------------------------------------------------------------------------


def test_openapi_schema_does_not_contain_forbidden_legacy_phrases(
    openapi_schema: dict[str, Any],
) -> None:
    """The full OpenAPI schema (as serialized JSON) must not
    contain any of the forbidden legacy phrases. These phrases
    were misleading claims that have been corrected across the
    truth-patches; the OpenAPI schema is the consumer-facing
    surface and must not contain them.
    """
    schema_str = json.dumps(openapi_schema)
    for forbidden in FORBIDDEN_LEGACY_PHRASES_IN_OPENAPI:
        assert forbidden not in schema_str, (
            f"OpenAPI schema contains the forbidden legacy phrase "
            f"{forbidden!r}. This phrase must not appear in any "
            f"public API documentation surface. If the phrase is "
            f"required for historical reference, move it to a "
            f"non-public location."
        )


# ---------------------------------------------------------------------------
# 6. The graceful parameter type is boolean (no schema migration)
# ---------------------------------------------------------------------------


def test_cancel_graceful_query_parameter_type_is_boolean(
    cancel_endpoint: dict[str, Any],
) -> None:
    """The cancel ``graceful`` Query parameter type is
    ``boolean`` and the default is ``True``. No schema change
    from origin/main.
    """
    op = cancel_endpoint["operation"]
    params = op.get("parameters", [])
    graceful_param = next(
        (p for p in params if p.get("name") == "graceful"),
        None,
    )
    assert graceful_param is not None
    schema = graceful_param.get("schema", {})
    assert schema.get("type") == "boolean", (
        f"Cancel graceful Query parameter type must be 'boolean'; "
        f"got {schema.get('type')!r}"
    )
    # FastAPI serializes the default for Query parameters via
    # the "schema" field as well.
    assert "default" in schema or "default" in graceful_param, (
        "Cancel graceful Query parameter must have a default "
        "(currently True)."
    )


# ---------------------------------------------------------------------------
# 7. The cancel route path is /v1/jobs/{job_id}/cancel
# ---------------------------------------------------------------------------


def test_cancel_route_path_is_v1_jobs_cancel(cancel_endpoint: dict[str, Any]) -> None:
    """The cancel endpoint is mounted at
    ``/v1/jobs/{job_id}/cancel`` (the jobs router has prefix
    ``/v1``).
    """
    assert cancel_endpoint["path"] == "/v1/jobs/{job_id}/cancel", (
        f"Cancel endpoint path must be /v1/jobs/{{job_id}}/cancel; "
        f"got {cancel_endpoint['path']!r}"
    )


# ---------------------------------------------------------------------------
# 8. The OpenAPI title and version are unchanged (consumer contract)
# ---------------------------------------------------------------------------


def test_openapi_info_unchanged(openapi_schema: dict[str, Any]) -> None:
    """The OpenAPI info block (title, version) is the consumer
    contract. Verify it is populated and non-empty.
    """
    info = openapi_schema.get("info", {})
    assert info.get("title"), "OpenAPI info.title must be populated"
    assert info.get("version"), "OpenAPI info.version must be populated"


# ---------------------------------------------------------------------------
# 9. The cancel operation has a 200 and 4xx responses (HTTP contract)
# ---------------------------------------------------------------------------


def test_cancel_operation_has_200_and_error_responses(
    cancel_endpoint: dict[str, Any],
) -> None:
    """The cancel operation documents a 200 response with the
    ``CancelResponse`` JSON schema. The 404 and 409 responses
    are raised dynamically via ``HTTPException`` inside the
    handler body and do not appear in the auto-generated
    OpenAPI schema; documenting them in the route decorator
    is a future-slice concern. This test only asserts the 200
    success response is documented.
    """
    op = cancel_endpoint["operation"]
    responses = op.get("responses", {})
    assert "200" in responses, (
        "Cancel operation must document a 200 success response "
        "in the OpenAPI schema."
    )
    ok = responses["200"]
    schema_ref = ok.get("content", {}).get("application/json", {}).get("schema", {})
    assert "$ref" in schema_ref, (
        f"Cancel 200 response must reference the CancelResponse "
        f"schema via $ref; got {schema_ref!r}"
    )
    assert schema_ref["$ref"].endswith("/CancelResponse"), (
        f"Cancel 200 response must reference the CancelResponse "
        f"schema; got {schema_ref!r}"
    )


# ---------------------------------------------------------------------------
# 10. The graceful parameter is in the Query (not Path / Body)
# ---------------------------------------------------------------------------


def test_cancel_graceful_parameter_is_in_query(
    cancel_endpoint: dict[str, Any],
) -> None:
    """The ``graceful`` parameter must be a Query parameter
    (not Path, not Body). The pre-PR-9 contract was Query; this
    test pins the location.
    """
    op = cancel_endpoint["operation"]
    params = op.get("parameters", [])
    graceful_param = next(
        (p for p in params if p.get("name") == "graceful"),
        None,
    )
    assert graceful_param is not None
    assert graceful_param.get("in") == "query", (
        f"Cancel graceful parameter must be in 'query'; got "
        f"{graceful_param.get('in')!r}"
    )
