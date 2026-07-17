"""File ID helper — Sprint 19 Slice 4 (§4.2).

`file_id` se deriva deterministamente del contenido del archivo
(SHA-256 truncado a 32 hex chars) para que toda la pipeline sea
idempotente. Single source of truth: este módulo.

Usado por:
- `hermes.memory.drop_watcher.DropWatcher.process_path` (§4 step 2)
- `hermes.memory.collections.VaultCollectionsRepo` M6 Phase 2 reconciliation
- `hermes.memory.ingest_router` (Sprint 17, lee manifest.id)
- API endpoints (file lookup by id)
"""

from __future__ import annotations

import hashlib
from pathlib import Path

#: Tamaño del chunk para leer el archivo sin cargar todo en memoria.
#: 256 KB es el sweet spot entre syscalls y memoria para archivos grandes
#: (PDFs de 50MB, JPGs de 10MB). Era 64KB en versiones anteriores; el bump
#: a 256KB es una mejora de performance trivial (Sprint 19 Slice 4b LLM
#: review 2026-07-10: Mistral Devstral 2512 flagged 64KB como "ineficiente
#: para archivos muy grandes"). No tocar sin benchmark.
_CHUNK_SIZE = 262_144  # 256 KB

#: Largo del file_id (32 hex chars = 128 bits, suficiente para
#: identificar unicos en escala "second brain" personal).
_FILE_ID_LENGTH = 32


def file_id_from_path(path: Path) -> str:
    """Devuelve el `file_id` canónico = SHA-256(path content)[:32].

    Argumentos:
        path: ruta al archivo en el filesystem. Se abre en 'rb'.

    Returns:
        str de 32 chars lowercase hex (e.g. "e3b0c44298fc1c149afbf4c8996fb924").

    Edge cases:
        - Archivo vacío: SHA-256 de bytes vacíos =
          "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
          → truncado a 32 chars: "e3b0c44298fc1c149afbf4c8996fb924".
          Todos los archivos vacíos comparten este ID — intencional, son
          contenido idéntico. La UNIQUE constraint sobre `vault_files.path`
          previene colisión en DB.
        - Symlinks: este helper NO resuelve symlinks por sí mismo. Hashea
          el contenido literal del path que se le pasa. Si el path es un
          symlink, se hashea el symlink mismo (típicamente
          "<symlink target path>"), no el target. Si querés el contenido
          del target, hace `path.resolve(strict=False)` ANTES de llamar.
          El DropWatcher (Sprint 19 §4) sí resuelve symlinks vía
          `Path.resolve()` ANTES de invocar este helper, como parte de
          su check de path traversal. Esa resolución ocurre en una
          capa distinta.
        - Concurrent writes: se hashea lo que esté presente al momento.
          El drop watcher tiene 100ms de debounce tras el evento `close`,
          así que el archivo debería estar completo. Si cambia entre
          hash y DB INSERT, el `file_id` es para el contenido leído —
          no corrupto, solo "snapshot de ese instante".
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()[:_FILE_ID_LENGTH]


def is_valid_file_id(s: str) -> bool:
    """True si `s` parece un file_id válido (32 lowercase hex chars)."""
    if len(s) != _FILE_ID_LENGTH:
        return False
    return all(c in "0123456789abcdef" for c in s)
