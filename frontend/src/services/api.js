/**
 * API service — single source of truth for all backend calls.
 *
 * Endpoints come from VITE_ENDPOINT_* in .env. Vite's dev proxy
 * forwards /api/* to VITE_BACKEND_URL so the backend host never
 * appears in the client bundle.
 */

const ENDPOINTS = {
  pipeline: import.meta.env.VITE_ENDPOINT_PIPELINE,
  detect:   import.meta.env.VITE_ENDPOINT_DETECT,
  emotion:  import.meta.env.VITE_ENDPOINT_EMOTION,
  spoof:    import.meta.env.VITE_ENDPOINT_SPOOF,
  verify:   import.meta.env.VITE_ENDPOINT_VERIFY,
  register: import.meta.env.VITE_ENDPOINT_REGISTER,
};

export class ApiError extends Error {
  constructor(message, status, body) {
    super(message);
    this.status = status;
    this.body = body;
    this.name = 'ApiError';
  }
}

/**
 * Generic image POST that returns BOTH the parsed body and timing/status info.
 * Used by the diagnostics page so we can show latency, raw response, etc.
 */
export async function postImageDetailed(endpoint, blob, extraFields = {}) {
  const formData = new FormData();
  for (const [k, v] of Object.entries(extraFields)) formData.append(k, v);
  formData.append('file', blob, 'frame.jpg');

  const start = performance.now();
  let resp, body, error = null;
  try {
    resp = await fetch(endpoint, { method: 'POST', body: formData });
  } catch (e) {
    return {
      ok: false,
      status: 0,
      latency: Math.round(performance.now() - start),
      body: null,
      error: e.message || 'Network error — backend not reachable',
    };
  }
  const latency = Math.round(performance.now() - start);

  try {
    body = await resp.json();
  } catch {
    body = null;
  }

  if (!resp.ok) {
    error = body?.detail || `HTTP ${resp.status}`;
  }
  return { ok: resp.ok, status: resp.status, latency, body, error };
}

/* ── Convenience wrappers for the diagnostics page ── */
export const postPipeline = (blob)       => postImageDetailed(ENDPOINTS.pipeline, blob);
export const postDetect   = (blob)       => postImageDetailed(ENDPOINTS.detect, blob);
export const postEmotion  = (blob)       => postImageDetailed(ENDPOINTS.emotion, blob);
export const postSpoof    = (blob)       => postImageDetailed(ENDPOINTS.spoof, blob);
export const postVerify   = (blob)       => postImageDetailed(ENDPOINTS.verify, blob);
export const postRegister = (blob, id)   => postImageDetailed(ENDPOINTS.register, blob, { person_id: id });

/* ── Simple wrappers (throw on error) used by the main page ── */
async function postOrThrow(endpoint, blob, extra) {
  const r = await postImageDetailed(endpoint, blob, extra);
  if (!r.ok) throw new ApiError(r.error || 'Request failed', r.status, r.body);
  return r.body;
}
export const analyzeFrame = (blob)       => postOrThrow(ENDPOINTS.pipeline, blob);
export const registerFace = (blob, id)   => postOrThrow(ENDPOINTS.register, blob, { person_id: id });

/** Health check */
export const ping = async () => {
  try {
    const r = await fetch('/docs', { method: 'HEAD' });
    return r.ok || r.status === 405;
  } catch { return false; }
};

/** Expose paths (for displaying in UI) */
export const getEndpointPaths = () => ({ ...ENDPOINTS });
