"""
PR Review via 3-tier parallel multi-provider cascade of free models.

Refactor 2026-07-08: el cascade secuencial previo (Phase 1 OpenRouter
sequential -> Phase 2 opencode-go sequential) se quedaba corto en PRs
donde varios modelos fallaban a la vez. Nuevo diseno:

TIER 1 (always fires, 4 modelos en paralelo):
  - OpenRouter: nemotron-3-ultra-550b-a55b:free   (Nemotron anchor)
  - OpenRouter: qwen/qwen3-coder:free             (Qwen, coder-native)
  - NVIDIA NIM: z-ai/glm-5.2                      (Zhipu, agentic)
  - Mistral:   devstral-2512                      (Mistral, coder-native)

TIER 2 (fires si Tier 1 dio < MIN_SUCCESSFUL_RESPONSES, 3 en paralelo):
  - NVIDIA NIM: meta/llama-3.3-70b-instruct       (Llama)
  - OpenCode Zen: deepseek-v4-flash-free          (DeepSeek)
  - OpenCode Zen: nvidia/nemotron-3-super-free    (Nemotron, 2nd provider)

TIER 3 (last-resort, fires si Tier 1+2 < MIN_SUCCESSFUL_RESPONSES, 4):
  - Mistral: codestral-2508
  - OpenCode Zen: north-mini-code-free
  - OpenCode Zen: mimo-v2.5-free
  - NVIDIA NIM: deepseek-ai/deepseek-v4-pro       (slow ~94s, last resort)

Cada tier corre TODOS sus modelos en paralelo via ThreadPoolExecutor.
Latencia p99 por tier ~= la respuesta mas lenta del tier (~6s para
Tier 1/2, ~94s si Tier 3 dispara con v4-pro).

Cross-family consensus: el compute_cross_family_consensus() agrupa
findings por (file, line, severity) y cuenta FAMILIAS DISTINTAS, no
modelos. Dos Nemotron variants que coinciden = 1 voto (mismo sesgo
subyacente), no 2. Esto evita inflar el consensus artificialmente.

Por que consensus es SIGNAL, no FILTER (the operator, 2026-07-08):
  - HIGH CONF (>=2 familias): cada modelo por separado puede alucinar
    (PR #68 gpt-oss 70% falsos positivos), pero si 2 familias distintas
    coinciden en el mismo finding es probablemente real.
  - UNCONFIRMED (1 familia): se surfacea igual con marca "requiere
    revision manual", no se descarta. La IA que maneja el codigo
    (humano + Mavis en sesiones futuras) es quien adjudica.
  - LGTM si no hay findings accionables.

Por que 3 tiers (no 2):
  - Tier 1 falla entero ~= raro pero pasa (PR #73: 5/5 OR modelos
    429 en una PR). Tier 2 anade providers distintos (NIM, OCZ) para
    cubrir si un provider entero esta caido.
  - Tier 3 anade last-resort reasoning (v4-pro) + diversity Mistral/OCZ
    para casos extremos. Hard stop despues de Tier 3: si todo falla,
    post manual-review notice, no infinite loop.

Cuotas (verificadas 2026-07-08):
  - OpenRouter :free: 20 req/min, 50 req/day COMPARTIDO entre todos
    los modelos :free. Tier 1 usa 2 (Nemotron Ultra + Qwen3 Coder) =
    25 PRs/dia maximo desde OR en happy path.
  - NVIDIA NIM: 40 req/min global por API key.
  - Mistral: per-model rate (devstral 0.83 r/s, codestral 2.08 r/s).
  - OpenCode Zen: free models, disponibilidad variable.

Privacy:
  - Solo el diff se envia, nunca el repo entero.
  - Paths sensibles se redactan antes del envio (SENSITIVE_PATH_PATTERNS).
  - Valores que parecen secrets tambien se redactan
    (SECRET_VALUE_PATTERNS): tokens OpenAI/Anthropic/NVIDIA/Mistral/
    Cerebras, GitHub PATs, Slack/Discord tokens, Telegram bot tokens,
    webhooks, JWTs.
  - Si el diff contiene > MAX_REDACTION_THRESHOLD secciones redactadas,
    se declina revisar y se postea un aviso de "manual review required"
    (should_skip_for_redactions). Defensiva en profundidad.
  - TODOS los free tiers pueden entrenar con datos. PRs con info
    sensible deben llevar el label `skip-llm-review` -- el workflow
    se salta silenciosamente sin invocar este script.

Usage:
  PR_REPO=owner/repo PR_NUMBER=42 GH_TOKEN=... \
  OPENROUTER_API_KEY=... [NVIDIA_API_KEY=...] [MISTRAL_API_KEY=...] \
  [OPENCODE_API_KEY=...] \
  python scripts/pr_review.py

Cualquier provider key puede faltar; los modelos de ese provider se
marcaran como "missing env" en el footer del comment y no consumen
cuota. Necesario al menos 1 provider key (sino el script retorna 2).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

OPENROUTER_URL = (
    "https://openrouter.ai/api/v1/chat/completions"  # legacy alias, see PROVIDERS['openrouter']
)
# Optional overrides via env vars. Defaults to the current public repo so
# OpenRouter dashboards look right out-of-the-box; if the repo is renamed
# (for example after a repository rename) set OPENROUTER_REFERER and
# OPENROUTER_TITLE accordingly to avoid pinning to a stale name in headers.
OPENROUTER_REFERER = os.environ.get("OPENROUTER_REFERER", "https://github.com/AdrianMedico/oroimen")
OPENROUTER_TITLE = os.environ.get("OPENROUTER_TITLE", "Oroimen PR Review")

# Optional OpenCode-compatible fallback endpoint. Operators may replace it
# without changing source; credentials remain in the environment.
OPENCODE_URL = os.environ.get("OPENCODE_BASE_URL", "https://opencode.ai/zen/v1/chat/completions")


# Provider registry: each provider has an OpenAI-compatible chat
# completions endpoint, the env var name holding the API key, and
# optional extra headers (e.g. OpenRouter attribution).
#
# Adding a new provider = add one entry here + ensure call_model() can
# reach its URL with Bearer auth. No further wiring needed; the tier
# configs below reference providers by string key.
PROVIDERS: dict[str, dict] = {
    "openrouter": {
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "env_key": "OPENROUTER_API_KEY",
        "extra_headers": {
            "HTTP-Referer": OPENROUTER_REFERER,
            "X-Title": OPENROUTER_TITLE,
        },
    },
    "nim": {
        # NVIDIA NIM free endpoint. 40 req/min GLOBAL cap (per API key).
        # Phone verification required to obtain the key.
        "url": "https://integrate.api.nvidia.com/v1/chat/completions",
        "env_key": "NVIDIA_API_KEY",
        "extra_headers": None,
    },
    "mistral": {
        # Mistral La Plateforme. Free tier (Experiment plan) opts into
        # data training -- same privacy caveat as OpenRouter free tier.
        "url": "https://api.mistral.ai/v1/chat/completions",
        "env_key": "MISTRAL_API_KEY",
        "extra_headers": None,
    },
    "opencode": {
        # Public OpenCode-compatible endpoint; override with OPENCODE_BASE_URL.
        "url": OPENCODE_URL,
        "env_key": "OPENCODE_API_KEY",
        "extra_headers": None,
    },
}


# Tier configs: list of (provider, model_id, label, family, timeout_secs).
# Family is used by compute_cross_family_consensus() to avoid counting
# two models from the same training family as two independent votes
# (e.g. two Nemotron variants agreeing = 1 vote, not 2).
#
# Dispatch (in main()):
#   - Tier 1 fires always (4 models in parallel, ~6s p99).
#   - Tier 2 fires only if Tier 1 returned < MIN_SUCCESSFUL_RESPONSES.
#   - Tier 3 fires only if Tier 1+2 combined < MIN_SUCCESSFUL_RESPONSES.
#   - After Tier 3, hard stop -- no further retries regardless of signal.
#
# Free-tier quotas (verified 2026-07-08 against cheahjs/free-llm-api-resources
# and individual provider dashboards):
#   - OpenRouter :free:  20 req/min, 50 req/day SHARED across all :free models.
#   - NVIDIA NIM:        40 req/min global per API key.
#   - Mistral:           per-model rate (e.g. devstral 0.83 req/s, codestral 2.08 req/s).
#   - OpenCode Zen:      free models only; availability varies.
#
# Happy path (Tier 1 fully succeeds) consumes 2 OR + 1 NIM + 1 Mistral quota.
TIER_1_MODELS: list[tuple[str, str, str, str, int]] = [
    # (provider, model_id, label, family, timeout_secs)
    (
        "openrouter",
        "nvidia/nemotron-3-ultra-550b-a55b:free",
        "NVIDIA Nemotron Ultra 550B",
        "nemotron",
        120,
    ),
    ("openrouter", "qwen/qwen3-coder:free", "Qwen3 Coder", "qwen", 90),
    ("nim", "z-ai/glm-5.2", "Z.AI GLM-5.2", "zhipu", 90),
    ("mistral", "devstral-2512", "Mistral Devstral 2512", "mistral", 90),
]

TIER_2_MODELS: list[tuple[str, str, str, str, int]] = [
    ("nim", "meta/llama-3.3-70b-instruct", "Llama 3.3 70B Instruct", "llama", 60),
    ("opencode", "deepseek-v4-flash-free", "DeepSeek v4 Flash (OCZ)", "deepseek", 90),
    ("opencode", "nvidia/nemotron-3-super-free", "NVIDIA Nemotron 3 Super (OCZ)", "nemotron", 90),
]

TIER_3_MODELS: list[tuple[str, str, str, str, int]] = [
    ("mistral", "codestral-2508", "Mistral Codestral 2508", "mistral", 60),
    ("opencode", "north-mini-code-free", "North Mini Code (OCZ)", "code-north", 90),
    ("opencode", "mimo-v2.5-free", "MiMo v2.5 (OCZ)", "mimo", 60),
    # v4-pro is slow (~94s) but reasoning-strong. Last-resort only.
    # Fires only if Tier 1 + Tier 2 both < MIN_SUCCESSFUL_RESPONSES.
    ("nim", "deepseek-ai/deepseek-v4-pro", "DeepSeek v4 Pro", "deepseek", 180),
]

# Minimum number of successful (non-None) responses before we stop
# escalating to the next tier. With 4 Tier-1 models, partial failure
# of 1 is normal -- 3 is a reasonable floor for any consensus signal.
MIN_SUCCESSFUL_RESPONSES = 3

# Paths que NUNCA deben salir del repo en el diff. Redacted a [REDACTED-PATH].
# Si tu repo tiene otros paths sensibles, añadirlos aqui.
SENSITIVE_PATH_PATTERNS: list[str] = [
    r"\.mavis/[^\s\n]+",
    r"\.ssh/[^\s\n]+",
    r"\.config/[^\s\n]+",
    r"\.aws/[^\s\n]+",
    r"\.kube/[^\s\n]+",
    r"\.npmrc",
    r"\.pypirc",
    r"\.netrc",
    r"\.env(\.[^\s\n]+)?",
    r"\.envrc",
    r"secrets?/[^\s\n]+",
    r"id_(rsa|ed25519|ecdsa|dsa)[^\s\n]*",
    r"[^\s\n]*\.pem",
    r"[^\s\n]*\.key",
    r"[^\s\n]*\.p12",
    r"[^\s\n]*\.pfx",
    r"hermes\.env",
    r"oroimen\.env",
    r"cookies\.txt",
    r"yt-cookies\.txt",
]

# Patrones de VALORES que parecen secrets. Si un PR mete accidentalmente
# un token en una linea (ej. "OPENAI_API_KEY=sk-abc123" en un .env), la
# sanitizacion de paths no lo protege -- el path se redacta pero el valor
# sigue. Esta segunda pasada cubre los formatos mas comunes.
# False positive rate bajo pero no cero: si rompe, anadir el caso y
# excluirlo explicitamente en el regex.
#
# Mantenida a 2026-07-08 con los providers free que usa el PR review
# cascade (OpenRouter + NVIDIA NIM + Mistral La Plateforme + Cerebras +
# opencode-go) y los que pueden aparecer en este repo (Telegram bot
# tokens, Slack/Discord webhooks).
SECRET_VALUE_PATTERNS: list[tuple[str, str]] = [
    # OpenAI (sk-...) / OpenRouter (sk-or-...) / Anthropic (sk-ant-...) /
    # OpenAI project keys (sk-proj-...). El guion interno esta permitido
    # porque project keys tienen '-' despues de 'proj'. OpenRouter usa el
    # mismo prefijo sk-.
    (r"sk-(?:proj-|ant-|or-)?[A-Za-z0-9_-]{20,}", "[REDACTED-SECRET]"),
    # GitHub tokens: classic (ghp_), OAuth (gho_), server (ghs_),
    # user-to-server (ghu_), refresh (ghr_). All are 36+ chars after prefix.
    (r"gh[opsru]_[A-Za-z0-9]{36,}", "[REDACTED-SECRET]"),
    # Slack tokens (xoxb- bot, xoxa- app-level, xoxp- user, etc.)
    (r"xox[baprs]-[A-Za-z0-9-]{10,}", "[REDACTED-SECRET]"),
    # Google API keys (AIza + 35 chars alfanumericos/guion/underscore)
    (r"AIza[A-Za-z0-9_-]{35}", "[REDACTED-SECRET]"),
    # AWS access key id (AKIA + 16 chars uppercase/num)
    (r"AKIA[A-Z0-9]{16}", "[REDACTED-SECRET]"),
    # NVIDIA NIM API keys (nvapi- + ~60 chars alfanumericos). Sin este
    # patron, un .env con NVIDIA_API_KEY=nvapi-... sale tal cual al LLM.
    (r"nvapi-[A-Za-z0-9]{20,}", "[REDACTED-SECRET]"),
    # Cerebras API keys (csk- + chars alfanumericos). Cerebras es fallback
    # Tier 2 del cascade review, mismo riesgo que NVIDIA si la key rota.
    (r"csk-[A-Za-z0-9]{20,}", "[REDACTED-SECRET]"),
    # Mistral La Plateforme API keys. Mistral no documenta el formato
    # publico; heuristica var-name para no false-positivear hashes/UUIDs
    # de 32 chars. Match: mistral_api_key=..., mistral-api-key: ..., etc.
    (r"(?i)\bmistral[_-]?api[_-]?key\s*[:=]\s*[A-Za-z0-9]{20,}", "[REDACTED-SECRET]"),
    # Telegram bot tokens: <bot_id>:<35-char-token>. CRITICO para este
    # proyecto (Oroimen es un bot de Telegram). Si alguien sube el token
    # por error en un commit, sale tal cual al LLM sin este patron.
    # Formato: 8-10 digit bot_id + ':' + 35 chars [A-Za-z0-9_-].
    (r"\b\d{8,10}:[A-Za-z0-9_-]{35}\b", "[REDACTED-SECRET]"),
    # Bearer / Authorization headers en texto plano
    (r"(\bBearer\s+)[A-Za-z0-9_.\-+/=]{20,}", r"\1[REDACTED-SECRET]"),
    # Slack incoming webhooks. URL completa (path-level redaction porque
    # el path de la URL identifica el workspace).
    (r"https://hooks\.slack\.com/services/[A-Za-z0-9_/-]+", "[REDACTED-PATH]"),
    # Discord webhooks (discord.com o discordapp.com, ambas validas).
    (r"https://(?:discord|discordapp)\.com/api/webhooks/[A-Za-z0-9_/-]+", "[REDACTED-PATH]"),
    # JSON Web Tokens (tres segmentos base64url separados por puntos).
    # El ultimo segmento (signature) puede ser corto, >= 1 char.
    # Antes requeria {10,} que fallaba con signatures de <10 chars.
    (r"eyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]+", "[REDACTED-JWT]"),
]

SYSTEM_PROMPT = """\
Eres un senior Python/infra engineer revisando un PR diff.

Foco (en orden de prioridad):
1. Correctness: bugs, race conditions, error handling gaps, off-by-one
2. Security: secret leaks, injection, insecure defaults, path traversal
3. Performance: O(n^2), blocking I/O en async paths, missing timeouts
4. Mantenibilidad: claridad, naming, tests faltantes para casos obvios

FORMATO DE SALIDA (obligatorio, JSON valido):
Responde UNICAMENTE con un objeto JSON (sin texto antes ni despues).
NO uses markdown, NO uses code fences (```), NO agregues prosa.

Schema esperado:
{
  "lgtm": <true | false>,
  "findings": [
    {
      "severity": "BLOCKING" | "SUGGESTION" | "NIT",
      "category": "correctness" | "security" | "performance" | "maintainability",
      "file": "<path relativo al repo>",
      "line": <int o null si no aplica>,
      "message": "<descripcion corta, max 1 frase>"
    }
  ]
}

Reglas:
- Si no encuentras nada accionable: {"lgtm": true, "findings": []}
- severity BLOCKING = must fix antes de mergear
- severity SUGGESTION = nice to have, no bloquea
- severity NIT = estilo/minor, ignorable
- Reporta TODOS los findings que encuentres, no solo el primero
- Cita line numbers exactos cuando sea posible
- Responde en espanol en el campo "message"\
"""


def get_diff(repo: str, pr_number: str) -> str:
    """Get the PR diff via gh CLI."""
    result = subprocess.run(
        ["gh", "pr", "diff", pr_number, "--repo", repo],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def sanitize_diff(diff: str) -> str:
    """Redact sensitive paths AND secret-like values.

    Two passes:
      1. Paths matching SENSITIVE_PATH_PATTERNS are replaced with [REDACTED-PATH].
      2. Inline values matching SECRET_VALUE_PATTERNS (OpenAI/GH/Slack/Google
         tokens, AWS keys, Bearer headers, JWTs) are replaced with
         [REDACTED-SECRET] / [REDACTED-JWT].

    Order matters: path redaction first, so values inside path strings still
    match the value patterns. (In practice values rarely appear inside
    paths, but it's defensive.)
    """
    out = diff
    for pattern in SENSITIVE_PATH_PATTERNS:
        out = re.sub(pattern, "[REDACTED-PATH]", out)
    for pattern, replacement in SECRET_VALUE_PATTERNS:
        out = re.sub(pattern, replacement, out)
    return out


# Sensibilidad del skip automatico. Si el diff tiene MAS redacciones que
# esto, NO enviamos al LLM — posteamos un "manual review required". Cubre
# el caso en el que las dos pasadas de sanitizacion dejan pasar algo
# raro (un secret en formato exotico) o cuando hay tantos paths sensibles
# tocados que es absurdo revisarlo. Ajustable por repo si hay falsos
# positivos legítimos (ej: un PR que renombra docenas de `*.env` refs).
MAX_REDACTION_THRESHOLD = int(os.environ.get("PR_REVIEW_REDACTION_THRESHOLD", "5"))


def count_redactions(sanitized_diff: str) -> int:
    """Count [REDACTED-*] occurrences after sanitize_diff().

    Returns the total number of redactions across all three markers:
    [REDACTED-PATH], [REDACTED-SECRET], [REDACTED-JWT]. Used to decide
    whether to skip automated review and request manual review instead.
    """
    return (
        sanitized_diff.count("[REDACTED-PATH]")
        + sanitized_diff.count("[REDACTED-SECRET]")
        + sanitized_diff.count("[REDACTED-JWT]")
    )


def should_skip_for_redactions(sanitized_diff: str, threshold: int) -> tuple[bool, int]:
    """Decide whether the sanitized diff has too many redactions to review.

    Returns (skip, count):
      - skip=True si count > threshold: defensiva en profundidad. Aunque
        las dos pasadas de sanitize_diff() cubran los formatos comunes,
        si la densidad de redacciones es anormal probablemente hay
        material sensible que no queremos enviar a un LLM externo.
      - skip=False si count <= threshold: revisable.

    Threshold por defecto 5 (configurable via PR_REVIEW_REDACTION_THRESHOLD).
    """
    count = count_redactions(sanitized_diff)
    return (count > threshold, count)


def call_model(
    model_id: str,
    label: str,
    timeout: int,
    sanitized_diff: str,
    api_key: str,
    *,
    endpoint: str = OPENROUTER_URL,
    extra_headers: dict[str, str] | None = None,
) -> str | None:
    """Call an OpenAI-compatible chat completions endpoint.

    Defaults to OpenRouter (Phase 1) with attribution headers. Pass a
    different `endpoint` + `extra_headers` to call opencode-go or any
    other OpenAI-compat provider. Returns assistant text or None on
    failure.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if extra_headers:
        headers.update(extra_headers)
    payload: dict[str, object] = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"PR diff:\n\n```diff\n{sanitized_diff}\n```",
            },
        ],
        "temperature": 0.2,
        "max_tokens": 2000,
    }
    try:
        resp = requests.post(
            endpoint,
            headers=headers,
            json=payload,  # type: ignore[arg-type]
            timeout=timeout,
        )
    except requests.Timeout:
        print(f"  [{label}] timeout after {timeout}s", file=sys.stderr)
        return None
    except requests.RequestException as e:
        print(f"  [{label}] network error: {e}", file=sys.stderr)
        return None

    if resp.status_code != 200:
        # Keep error body short to avoid leaking provider-side metadata
        # (and to keep our Actions logs tidy). 100 chars is plenty to
        # diagnose a 4xx vs 5xx without exposing internals.
        body = resp.text[:100].replace("\n", " ")
        print(f"  [{label}] HTTP {resp.status_code}: {body}", file=sys.stderr)
        return None

    try:
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except (KeyError, ValueError, IndexError) as e:
        print(f"  [{label}] parse error: {e}", file=sys.stderr)
        return None


CASCADE_COMMENT_MARKER = "## 🤖 PR Review"


def _find_existing_cascade_comment(repo: str, pr_number: str, gh_token: str) -> int | None:
    """Find a prior PR comment posted by github-actions[bot] whose body
    starts with CASCADE_COMMENT_MARKER.

    Returns the comment ID (int) if found, else None.

    Why we need this:
        Every push to a PR branch re-triggers the workflow. Concurrency
        cancels in-flight runs but already-posted comments persist. Without
        idempotency, a PR with N pushes ends up with N cascade comments —
        all of which look identical until the diff actually changes. This
        caused PR #103 (Sprint 16.8.1) to accumulate 5 nearly-identical
        BLOCKING claims before merge, all hallucinated.

    Implementation: `gh api .../comments --paginate` walks all pages via
    Link headers. We pass `-q` jq selector that narrows to bot comments
    whose body starts with the marker. The script then picks the highest
    ID (newest) as the upsert target.
    """
    env = {**os.environ, "GH_TOKEN": gh_token}
    query = (
        '.[] | select(.user.login == "github-actions[bot]")'
        f' | select(.body | startswith("{CASCADE_COMMENT_MARKER}")) | .id'
    )
    result = subprocess.run(
        [
            "gh",
            "api",
            f"repos/{repo}/issues/{pr_number}/comments",
            "--paginate",
            "--jq",
            query,
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return None
    ids = [int(line) for line in result.stdout.strip().split("\n") if line.strip().isdigit()]
    if not ids:
        return None
    return max(ids)


def post_comment(repo: str, pr_number: str, body: str, gh_token: str) -> None:
    """Upsert a comment via gh CLI: edit existing if bot already posted one in
    this PR with the cascade marker, otherwise post a new comment.

    Keeps the PR thread clean across multiple pushes (each push triggers a new
    cascade run; we want exactly one cascade comment per PR, updated in place).

    Falls back to creating a new comment if the lookup fails (network error,
    gh CLI timeout, etc.) — at worst we duplicate, never silently drop.
    """
    env = {**os.environ, "GH_TOKEN": gh_token}
    existing_id: int | None = None
    try:
        existing_id = _find_existing_cascade_comment(repo, pr_number, gh_token)
    except Exception as exc:
        print(f"[pr-review] WARNING: cascade comment lookup failed: {exc}", file=sys.stderr)

    if existing_id is not None:
        print(f"[pr-review] editing existing cascade comment {existing_id} in PR {pr_number}")
        result = subprocess.run(
            [
                "gh",
                "api",
                "-X",
                "PATCH",
                f"repos/{repo}/issues/comments/{existing_id}",
                "-f",
                f"body={body}",
            ],
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
        if result.returncode == 0:
            return
        # If PATCH failed (e.g., comment deleted by user between find and edit),
        # fall through to POST instead of raising — better to duplicate than drop.
        print(
            f"[pr-review] WARNING: PATCH failed (rc={result.returncode}); "
            f"falling back to POST. stderr: {result.stderr.strip()}",
            file=sys.stderr,
        )

    print(f"[pr-review] posting new cascade comment on PR {pr_number}")
    subprocess.run(
        [
            "gh",
            "api",
            "-X",
            "POST",
            f"repos/{repo}/issues/{pr_number}/comments",
            "-f",
            f"body={body}",
        ],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )


def parse_json_findings(response_text: str | None) -> tuple[bool, list[dict]] | None:
    """Parse the JSON-shaped review from an LLM response.

    The system prompt instructs models to reply with a strict JSON object
    of the form {"lgtm": bool, "findings": [...]}. We accept that shape,
    plus a few common deviations:

    1. Markdown code fences (```json ... ```) -- stripped before parsing.
    2. Prose wrapping a JSON object -- fallback regex extraction.
    3. Malformed JSON -- returns None so the caller can surface the
       raw text as a generic "unparsed" contribution.

    Returns:
        (lgtm, findings) on success; findings is a list of dicts (may be empty).
        None on parse failure.
    """
    if not response_text:
        return None
    text = response_text.strip()
    # Strip markdown fences if present.
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:-1] if lines and lines[-1].strip().startswith("```") else lines[1:]
        text = "\n".join(lines).strip()
    # Try strict JSON first.
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        # Fallback: extract first {...} that contains "findings".
        match = re.search(r'\{[^{}]*"findings"[^{}]*\[.*?\][^{}]*\}', text, re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group())
        except (json.JSONDecodeError, ValueError):
            return None
    if not isinstance(data, dict):
        return None
    lgtm = bool(data.get("lgtm", False))
    findings_raw = data.get("findings", [])
    if not isinstance(findings_raw, list):
        return None
    findings: list[dict] = [f for f in findings_raw if isinstance(f, dict)]
    # Normalize severity to UPPERCASE so the dedup key in
    # compute_cross_family_consensus treats "BLOCKING" (Mistral style) and
    # "blocking" (some models) as the same bucket. Without this, the same
    # finding surfaces twice in the PR comment (once in BLOCKING, once in
    # UNCONFIRMED) because the dedup key is case-sensitive.
    for f in findings:
        sev = f.get("severity")
        if isinstance(sev, str):
            f["severity"] = sev.strip().upper() or "SUGGESTION"
    return lgtm, findings


def compute_cross_family_consensus(
    all_responses: list[tuple],
) -> tuple[list, list, list[str]]:
    """Aggregate findings from all tier responses into consensus buckets.

    Each response tuple is:
        (provider, model_id, label, family, response_text, latency_secs, error)

    Consensus groups by (file, line, severity). The signal criterion is
    NUMBER OF DISTINCT FAMILIES, not number of models -- two Nemotron
    variants agreeing counts as 1 family vote (same underlying bias),
    not 2 (which would inflate confidence falsely).

    Returns:
        high_conf: list of (finding_key, contributors_list) where >=2
                   distinct families agree on this finding.
        unconfirmed: list of (finding_key, contributors_list) where only
                     1 family saw it (could be one model or several
                     models of the same family).
        lgtm_labels: list of model labels that returned lgtm=True with
                     no findings.
    """
    buckets: dict = defaultdict(lambda: {"families": set(), "contributors": []})
    lgtm_labels: list[str] = []

    for resp in all_responses:
        _provider, _model_id, label, family, response_text, _latency, _error = resp
        parsed = parse_json_findings(response_text)
        if parsed is None:
            continue
        lgtm, findings = parsed
        if lgtm and not findings:
            lgtm_labels.append(label)
            continue
        for f in findings:
            key = (
                str(f.get("file", "")),
                f.get("line"),
                str(f.get("severity", "SUGGESTION")),
            )
            buckets[key]["families"].add(family)
            buckets[key]["contributors"].append(
                (label, str(f.get("message", "")), str(f.get("category", "")))
            )

    high_conf = []
    unconfirmed = []
    for key, data in buckets.items():
        if len(data["families"]) >= 2:
            high_conf.append((key, data["contributors"]))
        else:
            unconfirmed.append((key, data["contributors"]))

    return high_conf, unconfirmed, lgtm_labels


def format_parallel_comment(
    high_conf: list,
    unconfirmed: list,
    lgtm_labels: list[str],
    all_responses: list[tuple],
) -> str:
    """Format the PR comment using cross-family consensus.

    Layout:
      - HIGH CONFIDENCE BLOCKINGs first (>=2 families agree) -- most signal.
      - UNCONFIRMED BLOCKINGs next (1 family saw it) -- requires manual
        review, but we don't hide them.
      - LGTM section if no actionable findings.
      - Footer with per-model attempt status (success/failure + latency).

    Project policy (2026-07-08): consensus is a SIGNAL, not a FILTER.
    Single-model findings surface as UNCONFIRMED rather than being
    silently dropped, so the human reviewer adjudicates.
    """
    successful = [r for r in all_responses if r[4]]
    parts: list[str] = []
    parts.append("## 🤖 PR Review (multi-model parallel)\n")
    parts.append(
        f"**{len(successful)}/{len(all_responses)}** modelos respondieron "
        f"(consensus cross-family sobre "
        f"{len({r[3] for r in successful})} familias distintas)\n"
    )
    parts.append("")

    if high_conf:
        parts.append("### 🚨 BLOCKING (consenso cross-family)\n")
        for (file, line, severity), contributors in high_conf:
            line_str = f"L{line}" if line is not None else "general"
            loc = f"`{file}` ({line_str})" if file else f"({line_str})"
            parts.append(f"- **{severity}** {loc}")
            for label, msg, cat in contributors:
                tag = f" [{cat}]" if cat else ""
                parts.append(f"    - _{label}_{tag}: {msg}")
        parts.append("")

    if unconfirmed:
        parts.append("### ⚠️ UNCONFIRMED (1 familia lo vio -- requiere revisión manual)\n")
        for (file, line, severity), contributors in unconfirmed:
            line_str = f"L{line}" if line is not None else "general"
            loc = f"`{file}` ({line_str})" if file else f"({line_str})"
            for label, msg, cat in contributors:
                tag = f" [{cat}]" if cat else ""
                parts.append(f"- {loc} ({severity}) -- _{label}_{tag}: {msg}")
        parts.append("")

    if not high_conf and not unconfirmed:
        parts.append("### ✅ LGTM\n")
        if lgtm_labels:
            parts.append(f"Modelos que firmaron LGTM: {', '.join(lgtm_labels)}")
        else:
            parts.append(
                "Ningún modelo devolvió findings accionables, pero tampoco "
                "firmaron LGTM explícitamente. Revisar respuestas crudas si "
                "hay dudas."
            )
        parts.append("")

    # Attempt summary footer.
    parts.append("<sub>")
    parts.append("Intentos por modelo (en orden de tier):")
    by_tier_order: list[tuple] = []
    for tier_name, tier_models in [
        ("Tier 1", TIER_1_MODELS),
        ("Tier 2", TIER_2_MODELS),
        ("Tier 3", TIER_3_MODELS),
    ]:
        by_tier_order.append((tier_name, tier_models))
    for tier_name, tier_models in by_tier_order:
        parts.append(f"  - {tier_name}:")
        for entry in tier_models:
            provider, model_id, label, family, _timeout = entry
            # Find matching response.
            match = next(
                (r for r in all_responses if r[0] == provider and r[1] == model_id),
                None,
            )
            if match and match[4]:
                latency = int(match[5])
                parts.append(f"      - OK  {label} (`{model_id}`, family={family}) -- {latency}s")
            else:
                err = match[6] if match else "not attempted"
                parts.append(f"      - FAIL {label} (`{model_id}`, family={family}) -- {err}")
    parts.append("")
    parts.append(
        "Diff sanitizado antes del envio (path patterns + secret value patterns). "
        "Privacy: todos los providers free tier pueden entrenar con los datos -- "
        "PRs sensibles deben usar el label `skip-llm-review`."
    )
    parts.append("</sub>")
    return "\n".join(parts)


def format_failure_comment(attempts: list[str]) -> str:
    """Posted when ALL models across ALL tiers failed (no provider
    responded at all). Each entry in `attempts` should be a single-line
    status marker per model. Format examples:
        "OK  Nemotron Ultra 550B (`nvidia/nemotron-3-...`) -- 5s"
        "FAIL Devstral 2512 (`devstral-2512`) -- no response (timeout)"
    """
    attempts_str = "\n".join(f"  - {a}" for a in attempts)
    return (
        f"## 🤖 PR Review (multi-model parallel)\n\n"
        f"⚠️ All review models failed across all 3 tiers "
        f"(rate limit, timeout, or missing API key). "
        f"Manual review recommended.\n\n"
        f"<sub>\n"
        f"Attempts (in tier order):\n{attempts_str}\n\n"
        f"Check the workflow logs for HTTP status codes per model. "
        f"Diff was sanitized before sending.\n"
        f"</sub>"
    )


def format_skip_comment(redaction_count: int, threshold: int) -> str:
    """Comment posted when LLM review is skipped due to too many redactions.

    This is the third "we did not silently pass" path: if the diff is
    dense with sensitive material even after sanitize_diff(), we tell
    the PR author and reviewer that automated review was declined.
    Exit code 0 (graceful); the script did its job by NOT sending
    sensitive content to a third-party LLM.
    """
    return (
        f"## 🤖 PR Review (LLM cascade)\n\n"
        f"⚠️ Skipped automated review: diff contains {redaction_count} "
        f"redacted sections (threshold {threshold}).\n\n"
        f"Even after sanitization, the volume of redactions suggests the "
        f"diff should not be reviewed by an external LLM. "
        f"**Manual review required.**\n\n"
        f"<sub>\n"
        f"Sanitization covers path patterns (`SENSITIVE_PATH_PATTERNS`) "
        f"and inline secret values (`SECRET_VALUE_PATTERNS`). If this "
        f"trigger is a false positive (e.g. a refactor across many env "
        f"files), raise `PR_REVIEW_REDACTION_THRESHOLD` as a repo "
        f"variable and re-run.\n"
        f"</sub>"
    )


def run_parallel_tier(
    models: list[tuple[str, str, str, str, int]],
    sanitized_diff: str,
    pr_meta: dict | None = None,
) -> list[tuple]:
    """Run all models in `models` concurrently and collect responses.

    Each model entry is (provider, model_id, label, family, timeout_secs).
    The provider determines the endpoint URL, API key env var, and any
    extra headers via PROVIDERS[provider]. All HTTP calls are issued via
    call_model(), which is provider-agnostic.

    Uses ThreadPoolExecutor because requests is sync and blocking I/O
    benefits from concurrency more than async would here (the GIL is
    released during socket waits). Max latency of a tier = the slowest
    model's response time (parallel), not the sum (sequential).

    Returns a list of result tuples, one per input model:
        (provider, model_id, label, family, response_text_or_None,
         latency_secs, error_string_or_None)

    response_text is None if the call failed (network/timeout/HTTP error
    or missing API key). error_string describes why for the footer.
    """

    def call_one(entry: tuple[str, str, str, str, int]) -> tuple:
        provider, model_id, label, family, timeout = entry
        prov_cfg = PROVIDERS.get(provider)
        if prov_cfg is None:
            return (provider, model_id, label, family, None, 0.0, f"unknown provider {provider!r}")
        api_key = os.environ.get(prov_cfg["env_key"], "").strip()
        if not api_key:
            return (
                provider,
                model_id,
                label,
                family,
                None,
                0.0,
                f"missing env {prov_cfg['env_key']}",
            )
        t0 = time.time()
        response = call_model(
            model_id=model_id,
            label=label,
            timeout=timeout,
            sanitized_diff=sanitized_diff,
            api_key=api_key,
            endpoint=prov_cfg["url"],
            extra_headers=prov_cfg.get("extra_headers"),
        )
        elapsed = time.time() - t0
        if response:
            return (provider, model_id, label, family, response, elapsed, None)
        return (
            provider,
            model_id,
            label,
            family,
            None,
            elapsed,
            "no response (timeout/HTTP error/network)",
        )

    results: list[tuple] = []
    if not models:
        return results
    with ThreadPoolExecutor(max_workers=len(models)) as executor:
        future_to_entry = {executor.submit(call_one, m): m for m in models}
        for future in as_completed(future_to_entry):
            try:
                results.append(future.result())
            except Exception as exc:  # defensive: call_one should not raise
                entry = future_to_entry[future]
                provider, model_id, label, family, _ = entry
                results.append(
                    (
                        provider,
                        model_id,
                        label,
                        family,
                        None,
                        0.0,
                        f"executor exception: {type(exc).__name__}: {exc}",
                    )
                )
    return results


def main() -> int:
    repo = os.environ.get("PR_REPO", "").strip()
    pr_number = os.environ.get("PR_NUMBER", "").strip()
    gh_token = os.environ.get("GH_TOKEN", "").strip()

    # Sanity-check the minimum env contract. We don't fail on missing
    # provider keys here -- run_parallel_tier() handles missing keys
    # gracefully per-model and marks them as failures. This way a
    # partially-configured repo (e.g. only OR + Mistral, no NIM) still
    # gets reviews from whatever providers ARE configured.
    missing = [
        k
        for k, v in [
            ("PR_REPO", repo),
            ("PR_NUMBER", pr_number),
            ("GH_TOKEN", gh_token),
        ]
        if not v
    ]
    if missing:
        print(f"ERROR: missing env vars: {missing}", file=sys.stderr)
        return 2

    # Quick health check: at least one provider's API key must be present,
    # otherwise the script would burn ~12 minutes discovering every model
    # has a missing key. If only PR_REPO/PR_NUMBER/GH_TOKEN are set and
    # NO provider key is set, fail fast.
    any_provider_key = any(os.environ.get(p["env_key"], "").strip() for p in PROVIDERS.values())
    if not any_provider_key:
        print(
            "ERROR: no provider API keys set "
            f"(need at least one of: {', '.join(p['env_key'] for p in PROVIDERS.values())})",
            file=sys.stderr,
        )
        return 2

    # Note: NO logueamos el nombre del repo. Ya aparece en el header del
    # run de Actions; reiterarlo en los logs del step solo añade superficie
    # sin valor.
    print(f"[pr-review] pr=#{pr_number}")
    print("[pr-review] fetching diff...")
    diff = get_diff(repo, pr_number)
    if not diff.strip():
        print("[pr-review] empty diff, nothing to review.")
        return 0

    print(f"[pr-review] diff size: {len(diff)} chars")
    sanitized = sanitize_diff(diff)
    skip, redaction_count = should_skip_for_redactions(sanitized, MAX_REDACTION_THRESHOLD)
    print(f"[pr-review] redactions: {redaction_count} (threshold {MAX_REDACTION_THRESHOLD})")
    if skip:
        print(
            f"[pr-review] skipping LLM review: {redaction_count} redactions "
            f"exceeds threshold {MAX_REDACTION_THRESHOLD}",
            file=sys.stderr,
        )
        post_comment(
            repo,
            pr_number,
            format_skip_comment(redaction_count, MAX_REDACTION_THRESHOLD),
            gh_token,
        )
        print("[pr-review] posted skip notice")
        return 0

    # ---- 3-tier parallel dispatch ----------------------------------------
    # Tier 1 fires always (4 models in parallel, ~6s p99).
    # Tier 2 fires only if Tier 1 returned < MIN_SUCCESSFUL_RESPONSES.
    # Tier 3 fires only if Tier 1+2 combined < MIN_SUCCESSFUL_RESPONSES.
    # After Tier 3, hard stop -- no further retries.

    pr_meta = {"repo": repo, "pr_number": pr_number}
    all_responses: list[tuple] = []
    started = time.time()

    print(f"[pr-review] Tier 1: firing {len(TIER_1_MODELS)} models in parallel...")
    tier1 = run_parallel_tier(TIER_1_MODELS, sanitized, pr_meta)
    all_responses.extend(tier1)
    t1_success = sum(1 for r in tier1 if r[4])
    t1_elapsed = int(time.time() - started)
    print(f"[pr-review] Tier 1 done: {t1_success}/{len(TIER_1_MODELS)} ok in {t1_elapsed}s")

    if t1_success < MIN_SUCCESSFUL_RESPONSES:
        print(
            f"[pr-review] Tier 1 short ({t1_success}/{MIN_SUCCESSFUL_RESPONSES}); "
            f"escalating to Tier 2 ({len(TIER_2_MODELS)} models)..."
        )
        tier2 = run_parallel_tier(TIER_2_MODELS, sanitized, pr_meta)
        all_responses.extend(tier2)
        t2_success = sum(1 for r in tier2 if r[4])
        t2_elapsed = int(time.time() - started)
        print(f"[pr-review] Tier 2 done: {t2_success}/{len(TIER_2_MODELS)} ok in {t2_elapsed}s")

        if (t1_success + t2_success) < MIN_SUCCESSFUL_RESPONSES:
            print(
                f"[pr-review] Tier 1+2 short ({t1_success + t2_success}/"
                f"{MIN_SUCCESSFUL_RESPONSES}); last-resort Tier 3 "
                f"({len(TIER_3_MODELS)} models)..."
            )
            tier3 = run_parallel_tier(TIER_3_MODELS, sanitized, pr_meta)
            all_responses.extend(tier3)
            t3_success = sum(1 for r in tier3 if r[4])
            t3_elapsed = int(time.time() - started)
            print(f"[pr-review] Tier 3 done: {t3_success}/{len(TIER_3_MODELS)} ok in {t3_elapsed}s")
            print("[pr-review] all tiers exhausted; hard stop (no further retries)")
        else:
            print("[pr-review] Tier 1+2 reached threshold; skipping Tier 3")
    else:
        print("[pr-review] Tier 1 reached threshold; skipping Tier 2/3")

    # ---- Consensus + comment ----------------------------------------------
    high_conf, unconfirmed, lgtm_labels = compute_cross_family_consensus(all_responses)
    total_success = sum(1 for r in all_responses if r[4])

    if total_success == 0:
        # All attempts failed across all tiers. Post the failure notice.
        print("[pr-review] all tiers failed; posting failure notice")
        # Build attempt markers for format_failure_comment.
        attempts: list[str] = []
        for _provider, model_id, label, _family, response, latency, error in all_responses:
            if response:
                attempts.append(f"OK {label} (`{model_id}`) -- {int(latency)}s")
            else:
                attempts.append(f"FAIL {label} (`{model_id}`) -- {error or 'unknown'}")
        post_comment(repo, pr_number, format_failure_comment(attempts), gh_token)
        return 1

    comment = format_parallel_comment(high_conf, unconfirmed, lgtm_labels, all_responses)
    post_comment(repo, pr_number, comment, gh_token)
    print(
        f"[pr-review] posted consensus comment: "
        f"{len(high_conf)} high-conf, {len(unconfirmed)} unconfirmed, "
        f"{len(lgtm_labels)} LGTM, {total_success}/{len(all_responses)} models ok"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
