"""Sprint 15 (US-3.1 §4 PR #69): embed_vault — ingesta una carpeta y embebe.

Escanea recursivamente un directorio (e.g. /data/Notes) y para cada
archivo:

1. Lee el contenido (texto plano, markdown; PDFs via pypdf).
2. Calcula SHA256 del texto extraido.
3. Busca dedup via `db.find_file_by_content_hash()` (PR #66).
   Si ya existe, no lo duplica. Solo verifica si tiene embedding
   (si no, lo embe).
4. Inserta/actualiza la fila en `files`.
5. Genera embedding y lo persiste en `file_embeddings` (PR #66).
6. Toca `last_referenced_at` (no aplica a ingestion, asi que skip).

Concurrencia limitada con `asyncio.Semaphore(5)` (Gemini 3.1 Pro §8.1):
un vault de 5K archivos no debe saturar OpenRouter (rate limit ~60 RPM
en free tier ni abrir 5K conexiones TCP simultaneas).

Caso de uso: el user tiene una carpeta de notas en su NAS y quiere que
Oroimen pueda buscarlas semanticamente. Corre `embed_vault.scan()` una
vez tras deploy, o periodicamente via cron de sistema.

NO es parte del flujo automatico de upload (/v1/files); eso ya embebe
inline (PR #66/67). embed_vault es para el caso "tengo 5K archivos
subidos por otras vias (Dropbox legacy, Drive en futuro, scp manual)
que quiero hacer buscables".
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hermes.services.embeddings import EmbeddingsService


logger = logging.getLogger(__name__)


# Extensions que sabemos extraer texto. PDF se extrae via pypdf.
# Markdown y texto plano se leen directo.
SUPPORTED_EXTENSIONS = {".md", ".txt", ".markdown", ".pdf"}

# Maximo tamano del archivo a procesar (bytes). Defiende contra
# "alguien metio un .iso de 4GB en la carpeta de notas".
# 10MB es mas que suficiente para una nota larga / paper PDF.
MAX_FILE_BYTES = 10 * 1024 * 1024  # 10 MB

# Timeout para la llamada a `embeddings_service.embed_and_store()` por
# archivo individual. Si el API cuelga (network stall, rate limit que
# no retorna 429, modelo saturado), sin este timeout el task queda
# pillado para siempre ocupando un slot del semaphore(5). Con timeout,
# el task se cancela tras `EMBED_TIMEOUT_S` segundos y el archivo cae
# en `failed` con `error_type=TimeoutError` para que el caller sepa
# que el archivo no se proceso y pueda reintentar mas tarde.
#
# 60s es generoso: embed_and_store en produccion tarda ~150ms (Qwen3
# embedding 8B sobre ~10KB texto). 60s cubre un reintento interno del
# backend + jitter de red. Si supera los 60s es porque algo esta
# realmente roto (no es variacion normal).
EMBED_TIMEOUT_S = 60.0


@dataclass
class VaultScanResult:
    """Resultado de un scan de embed_vault.

    Atributos publicos para que callers (CLI, API, tests) puedan
    inspeccionar que paso.
    """

    scanned: int  # archivos encontrados en el directorio
    skipped_unsupported: int  # extensiones no soportadas
    skipped_too_large: int  # archivos > MAX_FILE_BYTES
    skipped_empty: int  # texto extraido vacio (PDF escaneado sin OCR, txt vacio)
    skipped_unchanged: int  # ya tenian embedding (idempotente)
    embedded: int  # nuevos embeddings generados
    failed: int  # excepciones durante embed (network, pypdf, etc.)
    duration_s: float  # tiempo total del scan
    errors: list[tuple[str, str]]  # (filename, error_message)


async def _extract_text_async(path: Path) -> str:
    """Lee texto de un archivo. PDF via pypdf en threadpool.

    Razon del threadpool: pypdf es CPU-bound y bloquearia el event
    loop (mismo motivo que `_extract_file_text_async` en http_api).
    Para .txt/.md es IO puro (read() async-friendly via asyncio.to_thread
    para consistencia).
    """
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader

        def _read_pdf() -> str:
            reader = PdfReader(str(path))
            text_parts: list[str] = []
            for page in reader.pages:
                try:
                    text_parts.append(page.extract_text() or "")
                except Exception:
                    # Pagina corrupta: skip silencioso, no fallar el file entero.
                    continue
            return "\n".join(text_parts)

        return await asyncio.to_thread(_read_pdf)
    else:
        # txt, md, markdown: leemos bytes y decodificamos UTF-8
        # (con errors='replace' para no crashear con archivos exoticos).
        def _read_text() -> str:
            return path.read_bytes().decode("utf-8", errors="replace")

        return await asyncio.to_thread(_read_text)


async def scan(
    *,
    vault_path: Path,
    db: Any,  # Database
    embeddings_service: EmbeddingsService,
    progress_callback: Callable[[VaultScanResult], Awaitable[None]] | None = None,
    cancel_event: asyncio.Event | None = None,
) -> VaultScanResult:
    """Escanea `vault_path` recursivamente y embe archivos nuevos.

    Args:
        vault_path: directorio raiz a escanear. Si no existe, retorna
            resultado con `scanned=0` (no falla).
        db: Database instance (usa add_file, find_file_by_content_hash,
            add_file_embedding).
        embeddings_service: EmbeddingsService instance. Si RAG esta
            disabled, este metodo retorna VaultScanResult con
            `embedded=0` y un log warning — no falla.
        progress_callback: opcional, async callable invocado tras
            CADA archivo procesado. Util para CLI que imprime
            progreso, o API que reporta status. Si se pasa, se llama
            UNA VEZ por archivo (no por batch) para que el caller
            tenga granularity fina.
        cancel_event: opcional, asyncio.Event que el caller puede
            set() para cancelar el scan a mitad. Verificado entre
            archivos (no entre batches) — si el caller quiere
            cancelacion mas fina, debe implementar su propio loop.

    Returns:
        VaultScanResult con contadores y errores.

    Concurrencia:
        asyncio.Semaphore(5) limita a 5 archivos procesandose en
        paralelo. Esto evita saturar OpenRouter (free tier 60 RPM,
        5 concurrent = 5 RPM sustainable) y limita el uso de memoria
        (cada file se lee entero antes de embe).

    Performance:
        Vault 5K archivos x 10KB texto promedio x embed ~150ms/file
        / 5 concurrent = ~150s wall time. Aceptable para un job que
        corre 1x al deploy.
    """
    start = time.monotonic()
    result = VaultScanResult(
        scanned=0,
        skipped_unsupported=0,
        skipped_too_large=0,
        skipped_empty=0,
        skipped_unchanged=0,
        embedded=0,
        failed=0,
        duration_s=0.0,
        errors=[],
    )
    if not vault_path.exists() or not vault_path.is_dir():
        logger.warning(
            "embed_vault_path_missing",
            extra={"path": str(vault_path)},
        )
        result.duration_s = time.monotonic() - start
        return result

    if not getattr(embeddings_service, "is_enabled", False):
        logger.warning(
            "embed_vault_skipped_rag_disabled",
            extra={"path": str(vault_path)},
        )
        result.duration_s = time.monotonic() - start
        return result

    # Recoleccion de archivos (sync — rglob es CPU-light).
    candidates: list[Path] = []
    for path in vault_path.rglob("*"):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            result.skipped_unsupported += 1
            continue
        try:
            if path.stat().st_size > MAX_FILE_BYTES:
                result.skipped_too_large += 1
                continue
        except OSError:
            # File inaccesible (permisos, race condition). Skip.
            continue
        candidates.append(path)
    result.scanned = len(candidates)

    if not candidates:
        result.duration_s = time.monotonic() - start
        return result

    semaphore = asyncio.Semaphore(5)

    async def _process_one(path: Path) -> None:
        """Procesa un archivo. Aislado para que un fallo no tumbe el batch."""
        if cancel_event is not None and cancel_event.is_set():
            return
        async with semaphore:
            try:
                # 1. Extraer texto
                text = await _extract_text_async(path)
                if not text or not text.strip():
                    # Archivo vacio: skip silencioso (e.g. PDF escaneado
                    # sin OCR, .txt vacio, Markdown solo con headers).
                    # Contamos en `skipped_empty` para que el caller
                    # sepa que el archivo fue scanned pero no produjo
                    # texto util. Sin este counter, scanned > suma de
                    # los demas contadores (bug encontrado por Nemotron
                    # 3 Super review del PR #70).
                    result.skipped_empty += 1
                    return

                # 2. Content hash
                content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

                # 3. Dedup check
                existing = await db.find_file_by_content_hash(content_hash)
                file_id: str
                if existing is not None:
                    # Ya existe. Verificar si tiene embedding.
                    file_id = existing["id"]
                    has_emb = await db.get_file_embedding(file_id)
                    if has_emb is not None:
                        result.skipped_unchanged += 1
                        return
                    # Existe el file pero NO el embedding (e.g. file
                    # subido via /v1/files con RAG disabled en aquel
                    # momento). Re-embed.
                else:
                    # File nuevo. Insertar.
                    file_id = f"file_{content_hash[:24]}"  # usar prefix del
                    # hash como ID: estable entre re-scans, util para
                    # dedup futuro (si el archivo se mueve, el ID no
                    # cambia; el inode-style preview de PR #68 encaja
                    # aqui).
                    await db.add_file(
                        file_id=file_id,
                        filename=path.name,
                        mime_type="text/markdown"
                        if path.suffix.lower() in {".md", ".markdown"}
                        else ("application/pdf" if path.suffix.lower() == ".pdf" else "text/plain"),
                        size_bytes=path.stat().st_size,
                        extracted_text=text,
                        extraction_method="pypdf" if path.suffix.lower() == ".pdf" else "raw",
                        source="vault",
                        content_hash=content_hash,
                    )

                # 4. Embed (call a OpenRouter u otro backend).
                # El servicio se encarga de la persistencia + retry + log.
                # asyncio.wait_for protege contra cuelgues (network stall,
                # rate limit sin respuesta): sin esto, 5 archivos colgados
                # atascan el semaphore y el scan entero queda bloqueado.
                # 60s es generoso; embed_and_store real tarda ~150ms.
                # Chore 2026-07-05: Nemotron 3 Ultra 550B review.
                try:
                    embedded_ok = await asyncio.wait_for(
                        embeddings_service.embed_and_store(file_id, text),
                        timeout=EMBED_TIMEOUT_S,
                    )
                except TimeoutError:
                    logger.warning(
                        "embed_vault_embed_timeout",
                        extra={
                            "file_id": file_id,
                            # NOTA: 'filename' es reserved key en LogRecord
                            # de Python 3.11+. Usamos 'file_name' (igual
                            # patron que http_api.upload_file).
                            "file_name": path.name,
                            "timeout_s": EMBED_TIMEOUT_S,
                        },
                    )
                    result.failed += 1
                    result.errors.append((path.name, f"embed timeout after {EMBED_TIMEOUT_S}s"))
                    return
                if embedded_ok:
                    result.embedded += 1
                else:
                    result.failed += 1
                    result.errors.append((path.name, "embed_and_store returned False"))
            except Exception as exc:
                logger.exception(
                    "embed_vault_file_failed",
                    extra={"filename": path.name, "error_type": type(exc).__name__},
                )
                result.failed += 1
                result.errors.append((path.name, f"{type(exc).__name__}: {exc}"[:200]))
            finally:
                if progress_callback is not None:
                    try:
                        await progress_callback(result)
                    except Exception:
                        logger.exception("embed_vault_progress_callback_failed")

    # Lanzar tasks. asyncio.gather con return_exceptions=False (los
    # errores ya están capturados en _process_one). El gather await
    # todos los tasks (cancelled o no) antes de retornar.
    tasks = [asyncio.create_task(_process_one(p)) for p in candidates]
    try:
        await asyncio.gather(*tasks)
    finally:
        result.duration_s = time.monotonic() - start
        logger.info(
            "embed_vault_scan_done",
            extra={
                "scanned": result.scanned,
                "embedded": result.embedded,
                "skipped_unchanged": result.skipped_unchanged,
                "skipped_empty": result.skipped_empty,
                "failed": result.failed,
                "duration_s": round(result.duration_s, 2),
            },
        )
    return result
