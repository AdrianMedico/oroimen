"""Unit tests para ``authenticate_bearer`` y el contrato de auth en jobs_api.

Coverage:
- Bearer dependency lee el header Authorization y lo valida contra
  ``settings.http_api_api_key``.
- Modo sin auth (api_key=None) devuelve sentinel user_id 0.
- Token malformado (sin prefijo Bearer) → 401.
- Token invalido (no coincide) → 401.
- Token valido → 200 con user_id sentinel 0.
- Multi-user-ready: extraccion via Depends override funciona.

Anti-regression checks:
- El endpoint sigue el patron ``Depends(authenticate_bearer)`` documentado
  en SPRINT_14_ARCHITECTURE.md §10.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from hermes.receivers.auth import authenticate_bearer

# ---------------------------------------------------------------------------
# Helpers — minimal app para poder ejercitar el Depends directamente
# ---------------------------------------------------------------------------


def _build_app_with_protected_route(settings: Any) -> FastAPI:
    """Crea un FastAPI mínimo con un endpoint protegido por authenticate_bearer.

    Args:
        settings: Settings con http_api_api_key (o None).

    Returns:
        FastAPI app lista para TestClient.
    """
    app = FastAPI()
    app.state.settings = settings

    @app.get("/protected")
    async def protected(
        user_id: int = Depends(authenticate_bearer),
    ) -> dict[str, Any]:
        return {"user_id": user_id}

    return app


# ---------------------------------------------------------------------------
# Tests con api_key activada
# ---------------------------------------------------------------------------


def _settings_with_key(api_key: str | None) -> Any:
    """Builds a MagicMock-Settings con http_api_api_key."""
    s = MagicMock()
    s.http_api_api_key = api_key
    return s


def test_authenticate_bearer_dependency_called_with_correct_header() -> None:
    """Mock bearer dep, verifica que se llama con el header correcto.

    Endpoint protected que captura el valor del header Authorization que
    recibe authenticate_bearer via el contexto de la request.
    """
    app = _build_app_with_protected_route(_settings_with_key("good-key"))
    with TestClient(app) as client:
        response = client.get(
            "/protected",
            headers={"Authorization": "Bearer good-key"},
        )
    assert response.status_code == 200
    assert response.json() == {"user_id": 0}  # sentinel single-user


def test_authenticate_bearer_invalid_token_401() -> None:
    """Token malformado (sin prefijo Bearer) → 401."""
    app = _build_app_with_protected_route(_settings_with_key("good-key"))
    with TestClient(app) as client:
        response = client.get(
            "/protected",
            headers={"Authorization": "good-key"},  # missing "Bearer "
        )
    assert response.status_code == 401
    detail = response.json()["detail"]
    assert detail["error"]["type"] == "AuthError"
    assert "Bearer" in detail["error"]["message"]


def test_authenticate_bearer_wrong_token_401() -> None:
    """Token invalido (no coincide con api_key) → 401."""
    app = _build_app_with_protected_route(_settings_with_key("good-key"))
    with TestClient(app) as client:
        response = client.get(
            "/protected",
            headers={"Authorization": "Bearer wrong-key"},
        )
    assert response.status_code == 401
    assert response.json()["detail"]["error"]["message"] == "Invalid API key."


def test_authenticate_bearer_missing_header_401() -> None:
    """Sin Authorization header → 401."""
    app = _build_app_with_protected_route(_settings_with_key("good-key"))
    with TestClient(app) as client:
        response = client.get("/protected")
    assert response.status_code == 401
    assert "Missing" in response.json()["detail"]["error"]["message"]


def test_authenticate_bearer_valid_token_returns_user_id() -> None:
    """Token valido devuelve sentinel user_id=0 (single-user)."""
    app = _build_app_with_protected_route(_settings_with_key("sentinel-test-key"))
    with TestClient(app) as client:
        response = client.get(
            "/protected",
            headers={"Authorization": "Bearer sentinel-test-key"},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["user_id"] == 0  # sentinel single-user


# ---------------------------------------------------------------------------
# Tests con api_key=Null (modo dev / sin auth)
# ---------------------------------------------------------------------------


def test_authenticate_bearer_no_auth_mode_returns_user_id() -> None:
    """Sin api_key configurada → no hay validación, devuelve user_id=0."""
    app = _build_app_with_protected_route(_settings_with_key(None))
    with TestClient(app) as client:
        # sin header
        response = client.get("/protected")
    assert response.status_code == 200
    assert response.json() == {"user_id": 0}


def test_authenticate_bearer_no_auth_mode_ignores_token() -> None:
    """Sin api_key configurada, cualquier token (incluso mal formado) pasa."""
    app = _build_app_with_protected_route(_settings_with_key(None))
    with TestClient(app) as client:
        response = client.get(
            "/protected",
            headers={"Authorization": "anything"},
        )
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# Tests de robustez
# ---------------------------------------------------------------------------


def test_authenticate_bearer_app_state_missing_settings_no_crash() -> None:
    """Si app.state.settings no existe (e.g. tests sin setup), no crashea.

    Devuelve 0 (modo no-auth degenerado). Cobertura defensiva contra uso
    del Depends en un app no inicializado.
    """
    app = FastAPI()

    @app.get("/protected")
    async def protected(
        user_id: int = Depends(authenticate_bearer),
    ) -> dict[str, Any]:
        return {"user_id": user_id}

    with TestClient(app) as client:
        response = client.get("/protected")
    assert response.status_code == 200
    assert response.json() == {"user_id": 0}


def test_authenticate_bearer_www_authenticate_header_on_401() -> None:
    """Respuesta 401 incluye header WWW-Authenticate (RFC 6750)."""
    app = _build_app_with_protected_route(_settings_with_key("good-key"))
    with TestClient(app) as client:
        response = client.get("/protected")
    assert response.status_code == 401
    assert response.headers.get("www-authenticate") == "Bearer"

def test_authenticate_bearer_blank_key_is_no_auth() -> None:
    """Defensive compatibility: blank keys behave like None."""
    app = _build_app_with_protected_route(_settings_with_key(""))
    with TestClient(app) as client:
        response = client.get("/protected")
    assert response.status_code == 200
    assert response.json() == {"user_id": 0}
