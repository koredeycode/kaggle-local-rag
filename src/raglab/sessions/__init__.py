"""Session model and store (§6).

A session is a titled, isolated workspace owning its own documents, chunks,
embeddings, and eval runs — a *data* boundary, not a compute boundary. Models
are shared across sessions; data never is, and the store makes that structural
rather than a matter of discipline.
"""
