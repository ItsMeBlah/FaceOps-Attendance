import React, { useCallback, useEffect, useRef, useState } from 'react';
import { Camera, CameraOff, RotateCcw, Trash2, Upload, UserPlus } from 'lucide-react';
import { registerFaces } from '../services/api.js';

const REQUIRED_IMAGES = 5;

export default function RegistrationView({ camera, push, onRegistered }) {
  const uploadRef = useRef(null);
  const imagesRef = useRef([]);
  const nextImageIdRef = useRef(1);
  const [personName, setPersonName] = useState('');
  const [images, setImages] = useState([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    imagesRef.current = images;
  }, [images]);

  useEffect(() => () => {
    imagesRef.current.forEach((image) => URL.revokeObjectURL(image.preview));
  }, []);

  const addImages = useCallback((files, source) => {
    const remaining = REQUIRED_IMAGES - imagesRef.current.length;
    const accepted = Array.from(files)
      .filter((file) => file.type.startsWith('image/'))
      .slice(0, remaining)
      .map((blob) => ({
        id: `${Date.now()}-${nextImageIdRef.current++}`,
        blob,
        source,
        preview: URL.createObjectURL(blob),
      }));

    if (!accepted.length) return;
    setImages((current) => [...current, ...accepted]);
    setError('');
  }, []);

  const captureFace = useCallback(async () => {
    if (!camera.active) {
      setError('Turn on the camera before capturing a face.');
      return;
    }
    if (imagesRef.current.length >= REQUIRED_IMAGES) return;

    const blob = await camera.captureBlob();
    if (!blob) {
      setError('The camera is not ready. Hold still and try again.');
      return;
    }
    addImages([blob], 'camera');
  }, [addImages, camera]);

  const removeImage = useCallback((id) => {
    setImages((current) => {
      const removed = current.find((image) => image.id === id);
      if (removed) URL.revokeObjectURL(removed.preview);
      return current.filter((image) => image.id !== id);
    });
  }, []);

  const clearImages = useCallback(() => {
    setImages((current) => {
      current.forEach((image) => URL.revokeObjectURL(image.preview));
      return [];
    });
    setError('');
  }, []);

  const submit = useCallback(async () => {
    const name = personName.trim();

    if (!name) {
      setError('Name is required.');
      return;
    }
    if (images.length !== REQUIRED_IMAGES) {
      setError(`Capture or upload exactly ${REQUIRED_IMAGES} face images.`);
      return;
    }

    setBusy(true);
    setError('');
    try {
      const result = await registerFaces(images.map((image) => image.blob), name);
      push(`${name} registered with ID ${result.person_id}`, 'success', 6000);
      setPersonName('');
      clearImages();
      await onRegistered();
    } catch (requestError) {
      setError(requestError.message || 'Registration failed.');
    } finally {
      setBusy(false);
    }
  }, [clearImages, images, onRegistered, personName, push]);

  const remaining = REQUIRED_IMAGES - images.length;

  return (
    <main className="registration">
      <section className="registration__capture">
        <header className="camera__head">
          <div className="camera__head-title">
            <span className="camera__section-num">§ 01</span>
            <span className="camera__section-title">Registration Camera</span>
          </div>
          <span className="t-label">{images.length} / {REQUIRED_IMAGES} faces collected</span>
        </header>

        <div className="camera__viewport registration__viewport">
          <span className="camera__corner tr" />
          <span className="camera__corner bl" />
          <video
            ref={camera.videoRef}
            className="camera__video"
            autoPlay
            playsInline
            muted
            style={{ display: camera.active ? 'block' : 'none' }}
          />
          {camera.active && <div className="camera__rec">READY TO CAPTURE</div>}
          {!camera.active && (
            <div className="camera__off">
              <CameraOff className="camera__off-icon" />
              <div>
                <div className="camera__off-title">Camera dormant</div>
                <div className="camera__off-sub">Start the camera or upload face images</div>
              </div>
            </div>
          )}
        </div>

        <div className="registration__capture-actions">
          <button
            className={`btn ${camera.active ? 'btn--ghost' : 'btn--primary'}`}
            type="button"
            onClick={() => camera.active ? camera.stop() : camera.start()}
            disabled={busy}
          >
            <Camera />
            {camera.active ? 'Turn off camera' : 'Turn on camera'}
          </button>
          <button
            className="btn btn--accent"
            type="button"
            onClick={captureFace}
            disabled={!camera.active || remaining === 0 || busy}
          >
            <Camera />
            Capture face
          </button>
          <button
            className="btn btn--ghost"
            type="button"
            onClick={() => uploadRef.current?.click()}
            disabled={remaining === 0 || busy}
          >
            <Upload />
            Upload images
          </button>
          <input
            ref={uploadRef}
            className="registration__file-input"
            type="file"
            accept="image/*"
            multiple
            onChange={(event) => {
              addImages(event.target.files || [], 'upload');
              event.target.value = '';
            }}
          />
        </div>

        <div className="registration__slots">
          {Array.from({ length: REQUIRED_IMAGES }, (_, index) => {
            const image = images[index];
            return (
              <div key={image?.id || index} className={`registration__slot ${image ? 'filled' : ''}`}>
                {image ? (
                  <>
                    <img src={image.preview} alt={`Registration face ${index + 1}`} />
                    <span className="registration__slot-source">{image.source}</span>
                    <button
                      className="registration__remove"
                      type="button"
                      onClick={() => removeImage(image.id)}
                      disabled={busy}
                      title={`Remove face ${index + 1}`}
                    >
                      <Trash2 />
                    </button>
                  </>
                ) : (
                  <span className="registration__slot-empty">
                    <strong>{String(index + 1).padStart(2, '0')}</strong>
                    face required
                  </span>
                )}
              </div>
            );
          })}
        </div>
      </section>

      <aside className="registration__details">
        <section className="section">
          <header className="section__head">
            <div className="section__head-title">
              <span className="section__num">§ 02</span>
              <span className="section__title">Person Details</span>
            </div>
            <span className="section__aside">All fields checked before submission</span>
          </header>

          <label className="registration__label" htmlFor="registration-name">
            Name <span>required</span>
          </label>
          <input
            id="registration-name"
            className="registration__input"
            value={personName}
            onChange={(event) => setPersonName(event.target.value)}
            placeholder="e.g. Minh Cao"
            disabled={busy}
            autoComplete="off"
          />
        </section>

        <section className="registration__summary">
          <div>
            <span className="registration__summary-label">Collection status</span>
            <strong>{remaining === 0 ? 'Ready to register' : `${remaining} face${remaining === 1 ? '' : 's'} remaining`}</strong>
          </div>
          <div className="registration__summary-meter">
            <span style={{ width: `${(images.length / REQUIRED_IMAGES) * 100}%` }} />
          </div>
        </section>

        {error && <div className="registration__error">{error}</div>}

        <div className="registration__submit-actions">
          <button
            className="btn btn--ghost"
            type="button"
            onClick={clearImages}
            disabled={!images.length || busy}
          >
            <RotateCcw />
            Clear faces
          </button>
          <button
            className="btn btn--primary"
            type="button"
            onClick={submit}
            disabled={busy || images.length !== REQUIRED_IMAGES || !personName.trim()}
          >
            <UserPlus />
            {busy ? 'Registering...' : 'Register person'}
          </button>
        </div>
      </aside>
    </main>
  );
}
