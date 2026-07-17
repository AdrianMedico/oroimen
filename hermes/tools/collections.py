"""Agent tools para vault collections (Sprint 19 Slice 3).

5 tools registradas en ToolRegistry que envuelven VaultCollectionsRepo:

- list_collections: tree structure (filter archived, depth limit)
- create_collection: new collection (UNIQUE name)
- add_file_to_collection: link file<->collection (idempotent)
- remove_file_from_collection: unlink (idempotent, allow_last flag)
- move_file_to_collection: atomic move (add to target, then remove from source)

Las tools NO llaman al HTTP API; usan el repo directamente. El ToolRegistry
tiene `db` en el closure (ver hermes/tools/builtin.py:register_builtin_tools).

Error contract:
- CollectionNotFoundError: collection_id no existe
- DuplicateCollectionError: name ya existe (UNIQUE constraint)
- FileNotInCollectionError: remove_file sin allow_last y es el unico link
- PermissionError-like: levantado por remove_file cuando violates allow_last contract

El agent loop captura estos y los convierte a tool error response.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from hermes.memory.collections import (
    CollectionNotFoundError,
    VaultCollectionsRepo,
)

if TYPE_CHECKING:
    from hermes.memory.db import Database

logger = logging.getLogger(__name__)

# Cap tree size to prevent 2500-char budget overflow (Verif 2 M1).
# 200 nodes ~= 8-12K chars serialized -- enough to detect the truncation
# at the LLM boundary without breaking realistic PARA use cases (typically
# <100 collections). When truncated, response includes `truncated: True`
# and `node_count` so the LLM can decide to call again with depth=1.
_MAX_TREE_NODES = 200

# ===========================================================================
# list_collections
# ===========================================================================


async def list_collections(
    include_archived: bool = False,
    depth: int = 5,
    *,
    db: Database,
) -> dict:
    """Lista todas las collections como tree (default excluye archivadas).

    Args:
        include_archived: si True, incluye collections con archived=1.
        depth: max profundidad del tree (1-10, default 5).
        db: Database instance (inyectado por ToolRegistry closure).

    Returns:
        Dict con clave 'collections' (lista de root collections). Cada
        root tiene 'children' recursivo (hasta depth limit). Filtra
        archived por default.

    Raises:
        ValueError: si depth fuera de rango [1, 10].
    """
    if not 1 <= depth <= 10:
        raise ValueError(f"depth must be 1-10, got {depth}")
    repo = VaultCollectionsRepo(db)
    flat = await repo.list_collections(include_archived=include_archived)

    # Cap node count BEFORE tree building (Verif 2 M1).
    # If truncated, signal in response so LLM can re-query with depth=1.
    truncated = False
    node_count = len(flat)
    if node_count > _MAX_TREE_NODES:
        truncated = True
        flat = flat[:_MAX_TREE_NODES]
        logger.info(
            "list_collections_truncated",
            extra={"node_count": node_count, "cap": _MAX_TREE_NODES},
        )

    # Build tree via parent_collection_id map
    children_map: dict[str | None, list[Any]] = {}
    for c in flat:
        children_map.setdefault(c.parent_collection_id, []).append(c)

    def _node(c: Any, current_depth: int) -> dict:
        d: dict[str, Any] = {
            "collection_id": c.collection_id,
            "name": c.name,
            "parent_collection_id": c.parent_collection_id,
            "description": c.description,
            "archived": bool(c.archived),
            "children": [],
        }
        if current_depth < depth:
            for child in children_map.get(c.collection_id, []):
                d["children"].append(_node(child, current_depth + 1))
        return d

    roots = children_map.get(None, [])
    result: dict[str, Any] = {
        "collections": [_node(c, 1) for c in roots],
    }
    if truncated:
        result["truncated"] = True
        result["total_node_count"] = node_count
        result["returned_node_count"] = len(flat)
    return result


LIST_COLLECTIONS_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "include_archived": {
            "type": "boolean",
            "default": False,
            "description": "Incluir collections archivadas (soft-deleted via M6)",
        },
        "depth": {
            "type": "integer",
            "minimum": 1,
            "maximum": 10,
            "default": 5,
            "description": "Max profundidad del tree (1-10)",
        },
    },
    "required": [],
    "additionalProperties": False,
}


# ===========================================================================
# create_collection
# ===========================================================================


async def create_collection(
    name: str,
    parent_collection_id: str | None = None,
    description: str | None = None,
    *,
    db: Database,
) -> dict:
    """Crea una collection nueva. Path en filesystem NO se crea (eso lo hace
    el usuario via drag-drop en Finder/SMB, o el drop_watcher en Sprint 19 Slice 4).

    Args:
        name: nombre UNIQUE (1-200 chars). Se trimea whitespace.
        parent_collection_id: UUID hex de la collection padre (opcional).
        description: descripcion libre (opcional).
        db: Database instance.

    Returns:
        Dict con la collection creada (mismo shape que list_collections node).

    Raises:
        ValueError: si name vacio o excede 200 chars.
        CollectionNotFoundError: si parent_collection_id no existe.
        DuplicateCollectionError: si name ya esta tomado.
    """
    # Defense in depth: enforce maxLength=200 at tool layer too. The HTTP
    # API does this via Pydantic; tool layer catches LLM-side bypass.
    # JSON Schema `maxLength` is advisory — not enforced by LLMs.
    if not isinstance(name, str):
        raise ValueError(f"collection name must be a string, got {type(name).__name__}")
    # Strip whitespace BEFORE length check so whitespace-only 200-char
    # input fails here, not at the repo layer (adversarial R2 MINOR-1).
    stripped_name = name.strip()
    if not stripped_name or len(stripped_name) > 200:
        raise ValueError(
            f"collection name must be 1-200 chars (after strip), got {len(stripped_name)} chars"
        )
    repo = VaultCollectionsRepo(db)
    coll = await repo.create_collection(
        name=name,
        parent_collection_id=parent_collection_id,
        description=description,
    )
    return {
        "collection_id": coll.collection_id,
        "name": coll.name,
        "parent_collection_id": coll.parent_collection_id,
        "description": coll.description,
        "archived": bool(coll.archived),
    }


CREATE_COLLECTION_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "minLength": 1,
            "maxLength": 200,
            "description": "Nombre UNIQUE de la collection (1-200 chars)",
        },
        "parent_collection_id": {
            "type": ["string", "null"],
            "description": "UUID hex del padre (opcional, None = root)",
        },
        "description": {
            "type": ["string", "null"],
            "description": "Descripcion libre (opcional)",
        },
    },
    "required": ["name"],
    "additionalProperties": False,
}


# ===========================================================================
# add_file_to_collection
# ===========================================================================


async def add_file_to_collection(
    file_id: str,
    collection_id: str,
    *,
    db: Database,
) -> dict:
    """Vincula un file (ya en vault_files) a una collection.

    Idempotente: si el link ya existe, retorna con added=False (no raise).

    Args:
        file_id: UUID hex del vault_files row.
        collection_id: UUID hex de la collection destino.
        db: Database instance.

    Returns:
        Dict {collection_id, file_id, added: bool}.

    Raises:
        CollectionNotFoundError: si collection_id no existe.
        sqlite3.IntegrityError: si file_id no existe en vault_files (FK).
    """
    repo = VaultCollectionsRepo(db)
    added = await repo.add_file_to_collection(file_id=file_id, collection_id=collection_id)
    return {
        "collection_id": collection_id,
        "file_id": file_id,
        "added": added,
    }


ADD_FILE_TO_COLLECTION_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "file_id": {"type": "string", "description": "UUID hex del vault_file"},
        "collection_id": {
            "type": "string",
            "description": "UUID hex de la collection destino",
        },
    },
    "required": ["file_id", "collection_id"],
    "additionalProperties": False,
}


# ===========================================================================
# remove_file_from_collection
# ===========================================================================


async def remove_file_from_collection(
    file_id: str,
    collection_id: str,
    allow_last: bool = False,
    *,
    db: Database,
) -> dict:
    """Quita el link file<->collection. NO borra el file del vault.

    Idempotente: si el link no existe, retorna removed=False.

    Args:
        file_id: UUID hex del vault_file.
        collection_id: UUID hex de la collection origen.
        allow_last: si True, permite remover cuando es el unico link del
            file (file queda 'unfiled' - sigue buscable por texto/embeddings).
            Default False -> raises si es el ultimo link.
        db: Database instance.

    Returns:
        Dict {collection_id, file_id, removed: bool}.

    Raises:
        CollectionNotFoundError: si collection_id no existe.
        ValueError: si allow_last=False y es el ultimo link del file.
    """
    repo = VaultCollectionsRepo(db)
    # Pre-check: si el file tiene SOLO este link, raise (unless allow_last).
    # Si tiene 0 links (file nunca fue vinculado a ninguna collection),
    # el remove es no-op idempotente -- no raise.
    if not allow_last:
        existing_links = await repo.list_collections_for_file(file_id)
        if len(existing_links) == 1 and existing_links[0].collection_id == collection_id:
            raise ValueError(
                f"file {file_id!r} has only 1 collection link "
                f"({existing_links[0].name!r}); "
                f"pass allow_last=true to confirm unlinking"
            )
    removed = await repo.remove_file_from_collection(file_id=file_id, collection_id=collection_id)
    return {
        "collection_id": collection_id,
        "file_id": file_id,
        "removed": removed,
    }


REMOVE_FILE_FROM_COLLECTION_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "file_id": {"type": "string", "description": "UUID hex del vault_file"},
        "collection_id": {
            "type": "string",
            "description": "UUID hex de la collection origen",
        },
        "allow_last": {
            "type": "boolean",
            "default": False,
            "description": "Permitir remover el unico link del file",
        },
    },
    "required": ["file_id", "collection_id"],
    "additionalProperties": False,
}


# ===========================================================================
# move_file_to_collection
# ===========================================================================


async def move_file_to_collection(
    file_id: str,
    from_collection_id: str,
    to_collection_id: str,
    *,
    db: Database,
) -> dict:
    """Shortcut: add a destino, luego remove de origen.

    Atomicidad: si add a destino FALLA (collection no existe, FK violation),
    el remove de origen NUNCA corre -> file permanece en origen. No data loss.

    Si add OK pero remove falla (raro, race condition), file queda en AMBAS
    collections. Log warning. Caller puede cleanup con remove explicito.

    Args:
        file_id: UUID hex del vault_file.
        from_collection_id: UUID hex de la collection origen.
        to_collection_id: UUID hex de la collection destino.
        db: Database instance.

    Returns:
        Dict {file_id, from_collection_id, to_collection_id,
              added: bool, removed: bool, moved: bool}.
        - added=True si ADD a to inserto una row nueva (file no estaba en to).
        - removed=True si REMOVE from from borro una row (file estaba en from).
        - moved=True si REMOVE from source borró algo (file estaba en from).
          moved=False NO significa "nada pasó": ADD a to aún puede haber
          ocurrido (ver campo `added`).

        Si from == to: early-return, ningún DB write
          (added=removed=moved=False).

    Raises:
        CollectionNotFoundError: si from O to no existe.
        sqlite3.IntegrityError: si file_id no existe en vault_files (FK).
    """
    if from_collection_id == to_collection_id:
        # No-op, same collection
        return {
            "file_id": file_id,
            "from_collection_id": from_collection_id,
            "to_collection_id": to_collection_id,
            "added": False,
            "removed": False,
            "moved": False,
        }

    repo = VaultCollectionsRepo(db)

    # Pre-validate both collections exist (clean error message antes de
    # que add_file_to_collection lance su propio error generico).
    from_coll = await repo.get_collection(from_collection_id)
    if from_coll is None:
        raise CollectionNotFoundError(from_collection_id)
    to_coll = await repo.get_collection(to_collection_id)
    if to_coll is None:
        raise CollectionNotFoundError(to_collection_id)

    # ADD first. If this fails, file stays in source.
    added = await repo.add_file_to_collection(file_id=file_id, collection_id=to_collection_id)

    # REMOVE from source. If this fails (race), file is in both.
    # Log warning but return success - caller can decide.
    try:
        removed = await repo.remove_file_from_collection(
            file_id=file_id, collection_id=from_collection_id
        )
    except Exception as exc:
        logger.warning(
            "move_file_to_collection: remove failed after add succeeded",
            extra={
                "file_id": file_id,
                "from_collection_id": from_collection_id,
                "to_collection_id": to_collection_id,
                "exc_type": type(exc).__name__,
            },
        )
        removed = False

    return {
        "file_id": file_id,
        "from_collection_id": from_collection_id,
        "to_collection_id": to_collection_id,
        "added": added,
        "removed": removed,
        "moved": removed,
    }


MOVE_FILE_TO_COLLECTION_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "file_id": {"type": "string", "description": "UUID hex del vault_file"},
        "from_collection_id": {
            "type": "string",
            "description": "UUID hex de la collection origen",
        },
        "to_collection_id": {
            "type": "string",
            "description": "UUID hex de la collection destino",
        },
    },
    "required": ["file_id", "from_collection_id", "to_collection_id"],
    "additionalProperties": False,
}


# ===========================================================================
# register helper
# ===========================================================================


def register_collections_tools(
    registry: Any,  # ToolRegistry (forward ref para evitar import circular)
    *,
    db: Database,
) -> None:
    """Registra las 5 collections tools en el ToolRegistry.

    Args:
        registry: ToolRegistry donde registrar.
        db: Database instance (cerrada sobre todas las tools).

    Note:
        db se cierra sobre cada tool via closures _list_collections,
        _create_collection, etc. Esto evita acoplamiento al nombre del
        parametro en el JSON schema (que el LLM nunca vera).
    """

    async def _list_collections(
        include_archived: bool = False,
        depth: int = 5,
    ) -> dict:
        return await list_collections(
            include_archived=include_archived,
            depth=depth,
            db=db,
        )

    async def _create_collection(
        name: str,
        parent_collection_id: str | None = None,
        description: str | None = None,
    ) -> dict:
        return await create_collection(
            name=name,
            parent_collection_id=parent_collection_id,
            description=description,
            db=db,
        )

    async def _add_file_to_collection(
        file_id: str,
        collection_id: str,
    ) -> dict:
        return await add_file_to_collection(
            file_id=file_id,
            collection_id=collection_id,
            db=db,
        )

    async def _remove_file_from_collection(
        file_id: str,
        collection_id: str,
        allow_last: bool = False,
    ) -> dict:
        return await remove_file_from_collection(
            file_id=file_id,
            collection_id=collection_id,
            allow_last=allow_last,
            db=db,
        )

    async def _move_file_to_collection(
        file_id: str,
        from_collection_id: str,
        to_collection_id: str,
    ) -> dict:
        return await move_file_to_collection(
            file_id=file_id,
            from_collection_id=from_collection_id,
            to_collection_id=to_collection_id,
            db=db,
        )

    registry.register(
        "list_collections",
        _list_collections,
        description=(
            "Lista las collections del vault como tree jerarquico. "
            "Excluye archivadas por default (archived=1). Usar cuando "
            "el user pregunta 'que collections tengo?' o 'donde esta X?'."
        ),
        schema=LIST_COLLECTIONS_SCHEMA,
    )
    registry.register(
        "create_collection",
        _create_collection,
        description=(
            "Crea una collection nueva. Nombre debe ser UNIQUE. "
            "NO crea el directorio en filesystem (eso lo hace el usuario "
            "via Finder/SMB o el drop_watcher). Usar cuando el user dice "
            "'crea collection Proyectos_Nuevo' o similar."
        ),
        schema=CREATE_COLLECTION_SCHEMA,
    )
    registry.register(
        "add_file_to_collection",
        _add_file_to_collection,
        description=(
            "Vincula un file existente (vault_files) a una collection. "
            "Idempotente. Usar cuando el user dice 'anade este PDF a "
            "Proyectos' o 'asocia foo.md a 01_Proyectos_Activos'."
        ),
        schema=ADD_FILE_TO_COLLECTION_SCHEMA,
    )
    registry.register(
        "remove_file_from_collection",
        _remove_file_from_collection,
        description=(
            "Quita el link file<->collection. NO borra el file del vault. "
            "Por defecto falla si es el ultimo link del file (file quedaria "
            "'unfiled'); pasar allow_last=true para confirmar. Usar cuando "
            "el user dice 'saca foo.md de Proyectos' o 'desvincula X de Y'."
        ),
        schema=REMOVE_FILE_FROM_COLLECTION_SCHEMA,
    )
    registry.register(
        "move_file_to_collection",
        _move_file_to_collection,
        description=(
            "Shortcut atomico: add file a destino, luego remove de origen. "
            "Si add falla, remove no corre (file permanece en origen). "
            "Si from==to, no-op. Usar cuando el user dice 'mueve foo.md "
            "de Proyectos a Archivo' o 'reasigna X de A a B'."
        ),
        schema=MOVE_FILE_TO_COLLECTION_SCHEMA,
    )


# Re-export para tests + callers externos
__all__ = [
    "ADD_FILE_TO_COLLECTION_SCHEMA",
    "CREATE_COLLECTION_SCHEMA",
    "LIST_COLLECTIONS_SCHEMA",
    "MOVE_FILE_TO_COLLECTION_SCHEMA",
    "REMOVE_FILE_FROM_COLLECTION_SCHEMA",
    "add_file_to_collection",
    "create_collection",
    "list_collections",
    "move_file_to_collection",
    "register_collections_tools",
    "remove_file_from_collection",
]
