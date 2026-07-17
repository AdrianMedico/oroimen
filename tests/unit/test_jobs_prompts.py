"""Unit tests for hermes.jobs.prompts (Q2 verifier finding + JSON path).

Anti-regression checks (deliverable §Q2):
- ``_strip_json_thinking_block`` handles escaped quotes inside ``"thinking"`` value.
- ``_strip_json_thinking_block`` handles nested braces inside ``"thinking"`` value.
- Malformed / truncated JSON thinking blocks leave the text intact (no crash).
- Real-world case (``thinking`` + ``content`` co-existing) is stripped cleanly.
- ``sanitize_summary`` full pipeline: JSON-with-content path still works.

The Q2 finding was about the PREVIOUS regex ``\\{\\s*"thinking"\\s*:\\s*".*?"\\s*,?``
which broke on:
  1. ``"thinking": "long content with \\"escaped\\" quotes"`` —
     ``.*?"`` non-greedy cuts at the first ``"`` → thinking leak.
  2. ``"thinking": "value with {nested} braces"`` —
     ``.*?`` stops before the mid-string ``}`` → thinking leak.

The new parser walks the JSON with explicit brace + string-state tracking,
so both cases now strip cleanly.
"""

from __future__ import annotations

from hermes.jobs.prompts import (
    _find_matching_close_brace,
    sanitize_summary,
)

# =============================================================================
# _find_matching_close_brace — basic correctness
# =============================================================================


def test_find_matching_close_brace_simple_object() -> None:
    """``{}`` trivial case."""
    text = "{}"
    # Position of '{' is 0, matching '}' is 1.
    assert _find_matching_close_brace(text, 0) == 1


def test_find_matching_close_brace_nested_objects() -> None:
    """``{a: {b: {c: 1}}}`` — proper depth tracking."""
    text = '{"a": {"b": {"c": 1}}}'
    assert _find_matching_close_brace(text, 0) == len(text) - 1


def test_find_matching_close_brace_ignores_braces_in_strings() -> None:
    """``{"k": "v with {brace} inside"}`` — braces inside strings don't count."""
    text = '{"k": "v with {brace} inside"}'
    assert _find_matching_close_brace(text, 0) == len(text) - 1


def test_find_matching_close_brace_handles_escaped_quotes() -> None:
    """``{"k": "with \\"q\\" inside"}`` — backslash-escape keeps string open."""
    text = '{"k": "with \\"q\\" inside"}'
    # The closing '}' is the last char.
    assert _find_matching_close_brace(text, 0) == len(text) - 1


def test_find_matching_close_brace_returns_none_on_truncated() -> None:
    """Truncated ``{'thinking': 'unclosed string`` → returns None (no close)."""
    text = '{"thinking": "unclosed, never closes'
    assert _find_matching_close_brace(text, 0) is None


def test_find_matching_close_brace_returns_none_on_unbalanced() -> None:
    """``{a: {b: 1}`` — extra '{' never closed → returns None."""
    text = '{"a": {"b": 1}'
    assert _find_matching_close_brace(text, 0) is None


# =============================================================================
# _strip_json_thinking_block — the Q2 fix itself
# =============================================================================


def test_sanitize_summary_handles_json_with_escaped_quotes() -> None:
    """{\"thinking\": \"long content with \\\"escaped\\\" quotes\"} strips clean.

    Pre-fix: regex ``.*?`` cortaba en el primer ``"`` despues de
    ``escaped``, dejando ``leak here\"`` en el output. Post-fix: bracket-counter
    respeta el ``\\"`` dentro del string y consume el ``}`` final.
    """
    raw = (
        '{"thinking": "long content with \\"escaped\\" quotes here",\n'
        '"content": "real answer"}\n'
        "## Summary\nThe answer is 42."
    )
    cleaned = sanitize_summary(raw)
    assert "long content" not in cleaned
    assert "escaped" not in cleaned
    assert "leak" not in cleaned
    # Content del JSON o el texto trailing deben quedar.
    assert "## Summary" in cleaned or "real answer" in cleaned


def test_sanitize_summary_handles_json_with_nested_braces() -> None:
    """{\"thinking\": \"value with {nested} braces\"} strips clean.

    Pre-fix: ``.*?`` no-greedy se detenía en el primer ``}`` (el de ``nested``),
    dejando ``braces\"``` colgando. Post-fix: depth tracking ignora ``}``
    dentro de strings.
    """
    raw = (
        '{"thinking": "value with {nested} braces",\n' '"content": "ok"}\n' "## Summary\nAll good."
    )
    cleaned = sanitize_summary(raw)
    assert "nested" not in cleaned
    assert "## Summary" in cleaned
    assert "All good" in cleaned


def test_sanitize_summary_handles_malformed_json_gracefully() -> None:
    """Si no hay '}' balanceado, deja texto intacto (warning logged).

    Caso real: LLM truncó la respuesta mid-stream. El sanitizer NO debe
    hacer strip parcial ni crashear — mejor dejar el bloque visible que
    comer el summary entero.
    """
    raw = (
        '{"thinking": "this never closes, '
        "missing the closing brace.\n"
        "## Summary\nThe answer is 42."
    )
    cleaned = sanitize_summary(raw)
    # Texto preservado (no crash, no partial strip corrupto).
    assert "Summary" in cleaned
    assert "42" in cleaned
    # El '{' debe seguir ahí (parser defensivo: log warning + bail out).
    assert "{" in cleaned


def test_sanitize_summary_handles_truncated_json_gracefully() -> None:
    """Truncamiento puro (no '}' al final): no strip, no crash."""
    raw = '{"thinking": "truncated before end'
    # No exception raised — that's the contract.
    cleaned = sanitize_summary(raw)
    # Texto sigue ahí — no se hace drop silencioso.
    assert "truncated before end" in cleaned or "thinking" in cleaned


def test_sanitize_summary_real_world_dual_field_json() -> None:
    """Caso real del verifier: thinking + content, con escaped quotes en ambos.

    El LLM devuelve un objeto JSON con dos campos. ``sanitize_summary`` en
    Paso 1 (json.loads) detecta que es JSON completo, parsea, y devuelve
    SOLO el campo ``content``. No necesitamos la strip-brutal-force aquí.
    Verificamos que el pipeline produce el content correcto.
    """
    raw = (
        '{"thinking": "long internal reasoning with \\"quotes\\" and '
        '\\"more\\" and even {nested} braces that confused the regex",\n'
        '"content": "## Summary\\nReal answer here."}'
    )
    cleaned = sanitize_summary(raw)
    assert "long internal reasoning" not in cleaned
    assert "Real answer here" in cleaned
    assert "## Summary" in cleaned


def test_sanitize_summary_strip_block_not_full_json_when_no_content_field() -> None:
    """JSON sin 'content' field: parser strip-block se activa (defense in depth).

    Aqui simulamos el caso donde el JSON parsing falla (Paso 1) o el dict
    no tiene content — el strip-block via bracket-counter entra en juego.
    Construimos un texto donde el JSON está embebido en markdown (no es
    JSON puro del LLM, sino logs de error o recovery).
    """
    raw = (
        "Some prose before.\n"
        '{"thinking": "internal thoughts with \\"escaped\\" q", "id": "abc"}\n'
        "Some prose after."
    )
    cleaned = sanitize_summary(raw)
    # Bloque thinking debe estar gone.
    assert "internal thoughts" not in cleaned
    assert "escaped" not in cleaned
    # Texto circundante queda.
    assert "Some prose before" in cleaned
    assert "Some prose after" in cleaned


# =============================================================================
# sanitize_summary — full pipeline (regression for existing JSON extraction)
# =============================================================================


def test_sanitize_summary_full_json_with_content_extraction() -> None:
    """Regression: parsear JSON con `content` field debe devolver SOLO content.

    Test paralelo al ``test_sanitize_summary_handles_json_content_extraction``
    en tests/integration/test_jobs_service_phases.py — vive aquí también
    para unit-level coverage.
    """
    import json

    raw = json.dumps(
        {
            "thinking": "internal monologue here",
            "content": "## Summary\nThis is the actual response.",
        }
    )
    cleaned = sanitize_summary(raw)
    assert "internal monologue" not in cleaned
    assert "## Summary" in cleaned
    assert "This is the actual response" in cleaned


def test_sanitize_summary_strips_xml_thinking_tags() -> None:
    """Regression: los XML thinking blocks siguen funcionando (no-Q2 path)."""
    raw = (
        "Some intro.\n"
        "thought: should consider carefully\n"
        "thought done.\n"
        "## Summary\nThe answer is 42."
    )
    # El regex existente solo cubre variantes XML. Aqui solo verificamos
    # que la pipeline no rompe con texto llano.
    cleaned = sanitize_summary(raw)
    assert "## Summary" in cleaned
    assert "The answer is 42" in cleaned
