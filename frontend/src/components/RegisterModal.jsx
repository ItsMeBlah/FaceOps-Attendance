import React, { useEffect, useRef, useState } from 'react';

export default function RegisterModal({ open, onClose, onConfirm }) {
  const [name, setName] = useState('');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const inputRef = useRef(null);

  useEffect(() => {
    if (open) {
      setName('');
      setError('');
      setBusy(false);
      setTimeout(() => inputRef.current?.focus(), 100);
    }
  }, [open]);

  if (!open) return null;

  const submit = async () => {
    const trimmed = name.trim();
    if (!trimmed) {
      setError('A name is required for enrolment.');
      return;
    }
    setBusy(true);
    setError('');
    try {
      await onConfirm(trimmed);
    } catch (e) {
      setError(e.message || 'Enrolment failed.');
      setBusy(false);
    }
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal__num">§ Enrolment Form</div>
        <h2 className="modal__title">Enrol a new <em>face</em></h2>
        <p className="modal__body">
          Keep still for the camera.
        </p>

        {error && <div className="modal__error">// {error}</div>}

        <input
          ref={inputRef}
          className="modal__input"
          type="text"
          value={name}
          placeholder="Subject name or ID"
          onChange={(e) => setName(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter') submit(); }}
          disabled={busy}
          autoComplete="off"
        />

        <div className="modal__actions">
          <button className="btn btn--ghost" onClick={onClose} disabled={busy}>
            Cancel
          </button>
          <button className="btn btn--primary" onClick={submit} disabled={busy}>
            {busy ? 'Recording…' : 'Commit to registry'}
          </button>
        </div>
      </div>
    </div>
  );
}
