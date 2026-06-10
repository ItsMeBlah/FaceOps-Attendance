/**
 * API service — single source of truth for all backend calls.
 *
 * Endpoints come from VITE_ENDPOINT_* in .env. Vite's dev proxy
 * forwards /api/* to VITE_BACKEND_URL so the backend host never
 * appears in the client bundle.
 */

const ENDPOINTS = {
  pipeline:       import.meta.env.VITE_ENDPOINT_PIPELINE || '/api/pipeline/frame',
  registerBatch:  import.meta.env.VITE_ENDPOINT_REGISTER_BATCH || '/api/verification/register-batch',
  status:         import.meta.env.VITE_ENDPOINT_VERIFICATION_STATUS || '/api/verification/status',
};

export class ApiError extends Error {
  constructor(message, status, body) {
    super(message);
    this.status = status;
    this.body = body;
    this.name = 'ApiError';
  }
}

async function postImage(endpoint, blob, extraFields = {}) {
  const formData = new FormData();
  for (const [k, v] of Object.entries(extraFields)) {
    if (v !== undefined && v !== null && v !== '') formData.append(k, v);
  }
  formData.append('file', blob, 'frame.jpg');
  return doRequest(endpoint, formData);
}

/**
 * POST multiple images under a single key. Used for batch registration.
 */
async function postMultiImage(endpoint, blobs, extraFields = {}) {
  const formData = new FormData();
  for (const [k, v] of Object.entries(extraFields)) {
    if (v !== undefined && v !== null && v !== '') formData.append(k, v);
  }
  blobs.forEach((blob, i) => formData.append('files', blob, `frame_${i}.jpg`));
  return doRequest(endpoint, formData);
}

async function doRequest(endpoint, formData) {
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
  try { body = await resp.json(); } catch { body = null; }
  if (!resp.ok) error = body?.detail || `HTTP ${resp.status}`;
  return { ok: resp.ok, status: resp.status, latency, body, error };
}

async function postOrThrow(endpoint, blob, extra) {
  const r = await postImage(endpoint, blob, extra);
  if (!r.ok) throw new ApiError(r.error || 'Request failed', r.status, r.body);
  return r.body;
}
export const analyzeFrame = (blob)                      => postOrThrow(ENDPOINTS.pipeline, blob);

export const registerFaces = async (blobs, personName) => {
  const result = await postMultiImage(
    ENDPOINTS.registerBatch,
    blobs,
    { person_name: personName },
  );
  if (!result.ok) throw new ApiError(result.error || 'Registration failed', result.status, result.body);
  return result.body;
};

/* ── Status / health ─────────────────────────────────── */
export const getVerificationStatus = async () => {
  try {
    const r = await fetch(ENDPOINTS.status);
    if (!r.ok) return null;
    return await r.json();
  } catch { return null; }
};

export const ping = async () => {
  try {
    const r = await fetch('/docs', { method: 'HEAD' });
    return r.ok || r.status === 405;
  } catch { return false; }
};
