"""STT package: Speech-to-Text clients para Oroimen.

Estrategia actual: Gemini 3.1 Flash Lite (multimodal, free tier).

Por qué STT externo (no pasar audio directo al LLM):
- Sprint 12+ migración a MiniMax API (MiniMax-M3 multimodal nativo):
  el LLM actual YA procesaría audio directamente. Sin embargo, MANTENEMOS
  el STT externo por dos razones:
  1. Coste: Gemini Flash Lite es gratis vs $0.30/$0.60 per M tokens de
     MiniMax-M3 multimodal. Audio de voz (60s @ 24kbps) ≈ 180KB ≈ 45K
     tokens input — pasar directo al LLM costaría ~$0.014 por mensaje.
  2. Aislamiento de cuota: la cuota free de Gemini (500 RPD / 15 RPM)
     no acopla el audio al rate limit del chat principal.

Contexto histórico: originalmente bug opencode/opencode#30389 obligó a
externalizar el audio (mimo-v2.5 — único modelo de Go con capacidad de
audio — no procesaba input_audio via OpenCode Go). Ese bug ya no
aplica, pero la decisión de mantener STT externo se sostiene por las
dos razones de arriba.

Ver `06_Hermes_Asistente.md` sección 7.6 para detalles históricos.
"""
