"""Bearer auth dependency reusada por todos los routers v1.

Sprint 9.5 introdujo el middleware `_bearer_auth_middleware` dentro de
``create_app()`` para validar ``Authorization: Bearer <token>`` contra
``settings.http_api_api_key``. Sprint 14 (US-2.1) expone la misma logica
como una dependency FastAPI ``Depends``-friendly para que routers
montados via ``APIRouter`` (e.g. ``jobs_api``) puedan usarla sin
reinventar la rueda.

Contrato:
  - Si ``settings.http_api_api_key`` es ``None``: el endpoint es publico
    (legacy behavior, sin auth). Devuelve sentinel ``user_id=0``.
  - Si falta el header ``Authorization`` o no empieza por ``Bearer ``: 401.
  - Si el token no coincide con la API key configurada: 401.
  - Si OK: devuelve ``user_id=0`` (single-user setup). Multi-user ready:
    en el futuro el token puede llevar ``user_id`` codificado.

Mismo cuerpo de error que el middleware legacy para que los clientes que
ya consumen el HTTP API vean respuestas identicas independientemente
del router que sirva el endpoint (consistencia de UX).
"""

from __future__ import annotations

from typing import Annotated

from fastapi import HTTPException, Request, status

# Sentinel single-user. Multi-user en S15+ (TDD §user multi scope).
_SINGLE_USER_ID = 0


def authenticate_bearer(request: Annotated[Request, ...]) -> int:
    """FastAPI dependency: extrae user_id validando el Bearer token.

    Requiere ``app.state.settings`` (lo setea ``create_app()`` en
    ``hermes.receivers.http_api``). Si el atributo no existe, devolvemos
    0 silenciosamente (modo dev sin auth — solo aceptable en tests que
    no mockean settings).

    Returns:
        user_id int (single-user sentinel 0).

    Raises:
        HTTPException 401 si el header falta o el token no coincide.
    """
    settings = getattr(request.app.state, "settings", None)
    api_key = getattr(settings, "http_api_api_key", None) if settings else None
    if not api_key:
        # Modo no-auth (legacy / dev). Same as middleware short-circuit.
        return _SINGLE_USER_ID

    auth_header = request.headers.get("authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": {
                    "type": "AuthError",
                    "message": (
                        "Missing or invalid Authorization header. " "Expected: Bearer <token>."
                    ),
                }
            },
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = auth_header[len("Bearer ") :]
    if token != api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": {
                    "type": "AuthError",
                    "message": "Invalid API key.",
                }
            },
            headers={"WWW-Authenticate": "Bearer"},
        )
    return _SINGLE_USER_ID


__all__ = ["authenticate_bearer"]
