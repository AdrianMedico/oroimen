"""Static validation tests for Sprint 19 Slice 3 tool schemas.

Separated from test_collections_tools.py (which has pytestmark = asyncio)
because these tests are sync and would trigger pytest warnings if mixed in.
"""

from __future__ import annotations

from hermes.tools.collections import (
    ADD_FILE_TO_COLLECTION_SCHEMA,
    CREATE_COLLECTION_SCHEMA,
    LIST_COLLECTIONS_SCHEMA,
    MOVE_FILE_TO_COLLECTION_SCHEMA,
    REMOVE_FILE_FROM_COLLECTION_SCHEMA,
)


def test_schema_constants_have_required_keys() -> None:
    """Static check: schemas declarados en hermes.tools.collections son validos JSON Schema."""
    for schema in (
        LIST_COLLECTIONS_SCHEMA,
        CREATE_COLLECTION_SCHEMA,
        ADD_FILE_TO_COLLECTION_SCHEMA,
        REMOVE_FILE_FROM_COLLECTION_SCHEMA,
        MOVE_FILE_TO_COLLECTION_SCHEMA,
    ):
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "required" in schema
        assert "additionalProperties" in schema
