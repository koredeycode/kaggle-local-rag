# Security

Companion to §16. Threat model, boundaries, and what is deliberately *not* done.

## What this system is

A single operator running on their own Kaggle VM, reaching a FastAPI service over
a Cloudflare Quick Tunnel. Not a multi-tenant service. Not a production
deployment. There is no TLS of our own, no billing, no SLA, and sessions are
**workspaces, not security boundaries** — one operator token sees every session.

Stating that plainly matters more than any individual control below, because it
sets which threats are real. The realistic threats are: someone finds the tunnel
URL and tries to talk to the service; and an operator mistake exposes more than
intended. Accidental cross-session data contamination within one operator's own
data is not a security threat, which is why session isolation is about integrity
rather than confidentiality.

## The boundary

**Everything binds `127.0.0.1` by default.** This is the real boundary and it
holds, because the tunnel is the only ingress.

- The Cloudflare Quick Tunnel exposes **only** FastAPI. Never Ollama, never
  Jupyter, never a model dashboard.
- Ollama on `0.0.0.0` is an unauthenticated arbitrary-weight-execution endpoint
  on the public internet. Their old README's `OLLAMA_HOST=0.0.0.0` example is proof
  this is an easy mistake to make.

## Authentication

A locally generated runtime token. Three unauthenticated routes, and only three:

| Route | Why it is open |
|---|---|
| `GET /`, `/static/*` | the browser must load the client before it can authenticate |
| `GET /healthz` | readiness, used by the §19.1 cell-25 tunnel check |
| `POST /api/auth` | exchanges the operator token for a session cookie |

Everything else requires auth — bearer token or the cookie — **without exception**.
No exceptions for `/metrics`.

`POST /api/auth` is an *exchange*, not a mint. Returning the token you were just
given is a no-op loop that tells the client nothing it did not already have; what
it does is set a cookie the browser will actually send.

The token is **printed once, to the operator, by notebook cell 27**. It is never
written to disk, never logged, never persisted client-side beyond `sessionStorage`.
`Settings.runtime_token` has `repr=False` so a token cannot leak through a
traceback or a debugger transcript, and `from_env` deliberately refuses to read
`RAGLAB_RUNTIME_TOKEN` — otherwise `.env` would become a place the secret lives,
contradicting the rule.

The resolved rule is *"printed once to the operator, never stored, never logged"*,
not "never displayed". An operator has to be able to reach the system they just
built.

## A loopback check on `/metrics` is not a control

It is tempting to restrict `/metrics` to loopback. Do not. cloudflared runs *on
the VM* and connects to `127.0.0.1:<port>`, so **every tunnelled request appears to
arrive from loopback**. `client.host == "127.0.0.1"` passes for all public traffic
and can never fail once the tunnel is up.

That is a check which looks like protection and provides none. Auth is the
control; "binds 127.0.0.1" is the boundary. If a loopback restriction is ever
genuinely wanted it has to key on a forwarded header that cloudflared sets, not on
the socket peer address.

## Redaction

No prompts, no retrieved text, no tokens, no secrets in metrics or logs.

This is enforced in code rather than by convention, because conventions are what
get quietly broken:

- `logging.RedactionFilter` scrubs known-sensitive keys *and* free-form patterns:
  provider-shaped tokens, `Bearer` headers, `sha256:` document hashes, and Quick
  Tunnel hostnames. Every pattern has a matching test, because a redaction pattern
  that is never exercised is one that silently stopped matching after a refactor.
- `Span` validates its `attributes` mapping to scalars only, raising on anything
  else. A span buffer is the most likely place for retrieved text to leak: it is a
  debugging feature, and debugging features get pasted into issues.
- `scrub()` deliberately does *not* redact ordinary prose. Over-scrubbing is its
  own failure — a log that redacts everything is useless for debugging — and there
  is a test asserting `retrieve_ms=41 stage=retrieve` survives untouched.

## Tunnels

Quick Tunnels are public, random, temporary, and have no SLA. Cloudflare describes
them as a demo product with deliberate guardrails and recommends **named tunnels**
for anything real. This is stated in the README rather than implied away, and a
named tunnel is the first item in `production-migration.md`.

A named tunnel also removes the §19.3 streaming constraint entirely: the `GET`
buffering behaviour that forces POST-based SSE is a quick-tunnel property.

## Secrets in the notebook

Private clones use `GIT_ASKPASS` with a Kaggle Secret, never a token in the URL.
A URL token ends up in shell history, in `ps` output, and in the notebook's own
metadata.

## Mobile access

The tunnel URL only ever exists inside the notebook — the client is served
*through* that tunnel and is never told its own address. So the QR code is
rendered by **notebook cell 27 into the notebook output**, with the runtime token
printed beside it, rather than generated by the browser client.

That puts the QR where the operator actually is when they reach for the phone
(looking at Kaggle, not at the app on their laptop), removes a client dependency,
and leaves unanswered the question of how the URL would reach the client at all.

## Current limitations

- None of the routes exist yet; `ErrorCode` carries the full set and `STATE_TO_CODE`
  maps every §17.4 UI state to exactly one code, with a test asserting the mapping
  is complete. A missing mapping is then a startup failure rather than a runtime
  guess.
- The single-operator token model is assumed, not designed around. §24 Q2 asks for
  confirmation; if sessions ever need to be private to different people, this needs
  a real identity model and the current model is wrong, not merely insufficient.
- Auth is a shared bearer token. It has no expiry, no rotation, and no per-request
  scope. For a single operator over a temporary tunnel that is proportionate; it
  is not a production scheme and `production-migration.md` says so.
