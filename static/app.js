// Browser client — Phase 7. Not yet implemented.
//
// This file records the client-side contract so it is not rediscovered while
// writing the implementation. The API contract it will consume is decided (§16,
// §17) and the shapes below follow from it.

/**
 * Streaming transport (§19.3).
 *
 * Quick Tunnel buffers GET responses and flushes only when the connection
 * closes, and EventSource can only issue GET. So streaming is a POST carrying
 * standard SSE framing, read through fetch + ReadableStream.
 *
 * The framing is deliberately standard so a real deployment reverts to
 * EventSource by changing the method and nothing else.
 *
 * The alternative — buffer the whole answer server-side, then emit it in one
 * piece — was rejected: it gives up the stage readout that §17.5 calls the
 * product's honesty mechanism, and it is why the server buffers per stream-id
 * rather than withholding tokens from the client.
 */

/**
 * @param {string} sessionId
 * @param {string} question
 * @param {(event: {type: string, [k: string]: unknown}) => void} onEvent
 * @param {{lastEventId?: string}} [opts]
 * @returns {Promise<{aborted: boolean}>}
 */
export async function streamQuery(sessionId, question, onEvent, opts = {}) {
  const headers = {
    'Content-Type': 'application/json',
    Authorization: authHeader(),
  };
  if (opts.lastEventId) {
    // Reconnect after a phone backgrounding replays from the ring buffer instead
    // of restarting generation and paying for it twice (§17.6).
    headers['Last-Event-ID'] = opts.lastEventId;
  }

  const res = await fetch(`/api/sessions/${sessionId}/query?stream=true`, {
    method: 'POST',
    headers,
    body: JSON.stringify({ question }),
  });

  if (!res.ok) {
    // Every error uses one envelope; branch on `code`, never on `detail` (§16).
    const body = await res.json().catch(() => null);
    throw new Error(body?.error?.code ?? `http_${res.status}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      // SSE frames are separated by a blank line. Splitting on '\n\n' rather
      // than parsing line-by-line is what keeps a partial frame from being
      // dispatched twice.
      let split;
      while ((split = buffer.indexOf('\n\n')) !== -1) {
        const frame = buffer.slice(0, split);
        buffer = buffer.slice(split + 2);
        const event = parseFrame(frame);
        if (event) onEvent(event);
      }
    }
  } catch (err) {
    // Backgrounding a phone suspends the connection and the reader stalls or
    // errors. That is expected, not exceptional: keep the partial answer,
    // reconnect with Last-Event-ID, and re-enable input. Never lose the partial
    // answer — on a slow 8B model it may represent a minute of work (§17.6).
    if (err.name === 'AbortError') return { aborted: true };
    throw err;
  }
  return { aborted: false };
}

/** Parse one SSE frame. Returns null for comments such as `: heartbeat`. */
function parseFrame(frame) {
  let event = 'message';
  let data = '';
  for (const line of frame.split('\n')) {
    if (line.startsWith(':')) continue; // heartbeat
    if (line.startsWith('event:')) event = line.slice(6).trim();
    else if (line.startsWith('data:')) data += line.slice(5).trim();
  }
  if (!data) return null;
  try {
    return { type: event, ...JSON.parse(data) };
  } catch {
    return { type: event, raw: data };
  }
}

/**
 * Stage readout (§17.3). The gate gets its own row because it is plausibly the
 * slowest pre-generation stage on a T4 — a full LLM prefill over the assembled
 * candidates, not a matrix multiply. Folding it into "retrieving" would make a
 * multi-second pause look like a hang.
 */
export const STAGES = [
  { key: 'retrieving', label: 'Retrieving', note: 'dense, BM25, fusion' },
  { key: 'reranking', label: 'Reranking', note: 'cross-encoder scoring candidates' },
  { key: 'checking', label: 'Checking', note: 'relevance gate is grading the survivors' },
  { key: 'reading', label: 'Reading', note: 'generation has begun' },
];

/**
 * States that must exist as distinct renderings (§17.4). Not a state enum to
 * collapse — "no evidence found", "retrieval error", and "generation error"
 * rendering alike makes a working system look broken.
 */
export const STATES = [
  'index_empty',
  'indexing',
  'index_component_mismatch',
  'query_streaming',
  'no_evidence',
  'abstained',
  'citation_broken',
  'retrieval_error',
  'generation_error',
  'offline',
  'token_invalid',
  'no_sessions',
  'session_loading',
  'session_deleted',
  'session_title_empty',
];

/**
 * Token storage (§16, §17.2). `sessionStorage`, never `localStorage`: the token
 * must not outlive the tab. Never logged, never written to disk.
 */
export function authHeader() {
  const token = sessionStorage.getItem('raglab.token');
  return token ? `Bearer ${token}` : '';
}

export function setToken(token) {
  sessionStorage.setItem('raglab.token', token);
}

export function clearToken() {
  sessionStorage.removeItem('raglab.token');
}
