import React, { useState } from 'react';

/**
 * Single endpoint test card. Renders:
 *  - Title + path
 *  - Status pill (idle / busy / ok / err) + latency
 *  - Optional input fields (e.g. person_id for register)
 *  - "Run test" button
 *  - Result summary (parsed fields) + raw JSON
 */
export default function EndpointCard({
  title,
  method = 'POST',
  path,
  description,
  inputs = [],                 // [{ key, label, placeholder, required }]
  onRun,                       // (inputs) => Promise<{ok, status, latency, body, error}>
  hint,                        // small text next to the button
  summaryRenderer,             // (body) => array of {key, val, kind}
}) {
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState(null);
  const [values, setValues] = useState(() =>
    Object.fromEntries(inputs.map((i) => [i.key, i.default || '']))
  );

  const state = busy ? 'busy' : result?.ok ? 'ok' : result ? 'err' : 'idle';

  const statusText = {
    idle: 'Idle',
    busy: 'Running…',
    ok: `200 OK`,
    err: result ? `${result.status || 'ERR'}` : 'Error',
  }[state];

  const handleRun = async () => {
    // Validate required inputs
    for (const inp of inputs) {
      if (inp.required && !values[inp.key]?.trim()) {
        setResult({ ok: false, status: 0, latency: 0, error: `Missing: ${inp.label}` });
        return;
      }
    }
    setBusy(true);
    setResult(null);
    try {
      const r = await onRun(values);
      setResult(r);
    } catch (e) {
      setResult({ ok: false, status: 0, latency: 0, error: e.message });
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className={`endpoint-card ${state}`}>
      <div className="endpoint-card__head">
        <div>
          <div className="endpoint-card__title">{title}</div>
          <div className="endpoint-card__path">
            <span className="method">{method}</span>
            {path}
          </div>
        </div>
        <div className="endpoint-card__status">
          <span className={`endpoint-card__status-text ${state === 'idle' ? '' : state}`}>
            {statusText}
          </span>
          {result?.latency != null && state !== 'idle' && state !== 'busy' && (
            <span className="endpoint-card__latency">{result.latency}ms</span>
          )}
        </div>
      </div>

      <div className="endpoint-card__body">
        {description && <p className="endpoint-card__desc">{description}</p>}

        <div className="endpoint-card__actions">
          {inputs.map((inp) => (
            <input
              key={inp.key}
              className="endpoint-card__input"
              placeholder={inp.placeholder || inp.label}
              value={values[inp.key]}
              onChange={(e) => setValues((v) => ({ ...v, [inp.key]: e.target.value }))}
              disabled={busy}
            />
          ))}
          <button
            className="btn btn--accent btn--sm"
            onClick={handleRun}
            disabled={busy}
          >
            {busy ? 'Running…' : 'Run test'}
          </button>
          {hint && <span className="endpoint-card__hint">{hint}</span>}
        </div>

        {result?.error && (
          <div className="endpoint-card__result err">
            {result.error}
            {result.body && '\n\n' + JSON.stringify(result.body, null, 2)}
          </div>
        )}

        {result?.ok && result.body && summaryRenderer && (
          <div className="endpoint-card__summary">
            {summaryRenderer(result.body).map((s, i) => (
              <div key={i} className="endpoint-card__summary-item">
                <span className="endpoint-card__summary-key">{s.key}</span>
                <span className={`endpoint-card__summary-val ${s.kind || ''}`}>{s.val}</span>
              </div>
            ))}
          </div>
        )}

        {result?.ok && result.body && (
          <details style={{ marginTop: 10 }}>
            <summary style={{ cursor: 'pointer', fontFamily: 'var(--font-mono)', fontSize: 10, letterSpacing: '0.14em', textTransform: 'uppercase', color: 'var(--paper-3)' }}>
              Raw response
            </summary>
            <div className="endpoint-card__result">
              {JSON.stringify(result.body, null, 2)}
            </div>
          </details>
        )}
      </div>
    </div>
  );
}
