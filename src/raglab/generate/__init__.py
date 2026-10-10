"""Generation and citation validation (§14).

The Ollama client talks to the native `/api/chat`, because the OpenAI-compatible
route accepts neither `num_ctx` nor `think` and choosing it silently invalidates
the VRAM budget. `cite.py` resolves `[Sn]` markers back to chunks and catches the
three failure modes: hallucinated markers, over-citation, and unsupported claims.
"""
