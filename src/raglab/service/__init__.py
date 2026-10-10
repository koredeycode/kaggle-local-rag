"""FastAPI service (§16).

Every route touching corpus, index, query, or search state is nested under a
session id in the URL path, so the isolation guarantee is checkable by reading
the routes rather than auditing the store. Auth is required on everything except
the static client, `/healthz`, and the token exchange.
"""
