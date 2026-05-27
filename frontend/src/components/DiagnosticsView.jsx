import React, { useEffect, useRef, useState, useCallback } from 'react';
import EndpointCard from './EndpointCard.jsx';
import { PlayIcon, PauseIcon, CameraOffIcon } from './Icons.jsx';
import { HARDCODED_BOX, cropToBlob, frameToBlob } from '../utils/cropBox.js';
import {
  postPipeline, postDetect, postEmotion, postSpoof,
  postVerify, postRegister, ping, getEndpointPaths,
  getVerificationStatus,
} from '../services/api.js';

const PATHS = getEndpointPaths();

export default function DiagnosticsView({ camera, onRegistryChange }) {
  const overlayRef = useRef(null);
  const containerRef = useRef(null);
  const [pixelBox, setPixelBox] = useState(null);
  const [lastCropPreview, setLastCropPreview] = useState(null);
  const [backendReachable, setBackendReachable] = useState(null);
  const [verificationStatus, setVerificationStatus] = useState(null);

  // Backend reachability ping + status refresh
  useEffect(() => {
    let cancelled = false;
    const check = async () => {
      const ok = await ping();
      if (cancelled) return;
      setBackendReachable(ok);
      if (ok) {
        const status = await getVerificationStatus();
        if (!cancelled) setVerificationStatus(status);
      } else {
        setVerificationStatus(null);
      }
    };
    check();
    const t = setInterval(check, 5000);
    return () => { cancelled = true; clearInterval(t); };
  }, []);

  // Resize overlay canvas to container
  useEffect(() => {
    const resize = () => {
      const c = overlayRef.current;
      const box = containerRef.current?.getBoundingClientRect();
      if (!c || !box) return;
      c.width = box.width;
      c.height = box.height;
      drawOverlay();
    };
    resize();
    window.addEventListener('resize', resize);
    return () => window.removeEventListener('resize', resize);
  }, [camera.active]);

  // Hardcoded box drawn for emotion/spoof crop testing
  const drawOverlay = useCallback(() => {
    const c = overlayRef.current;
    if (!c) return;
    const ctx = c.getContext('2d');
    ctx.clearRect(0, 0, c.width, c.height);
    if (!camera.active) return;
    const v = camera.videoRef.current;
    if (!v || !v.videoWidth) return;

    const iw = v.videoWidth, ih = v.videoHeight;
    const cw = c.width, ch = c.height;
    const va = iw / ih, ca = cw / ch;
    let scale, ox = 0, oy = 0;
    if (ca > va) { scale = cw / iw; oy = (ch - ih * scale) / 2; }
    else         { scale = ch / ih; ox = (cw - iw * scale) / 2; }

    let bx = HARDCODED_BOX.x * iw * scale + ox;
    const by = HARDCODED_BOX.y * ih * scale + oy;
    const bw = HARDCODED_BOX.w * iw * scale;
    const bh = HARDCODED_BOX.h * ih * scale;
    bx = cw - bx - bw; // mirror

    setPixelBox({
      x: Math.floor(HARDCODED_BOX.x * iw),
      y: Math.floor(HARDCODED_BOX.y * ih),
      w: Math.floor(HARDCODED_BOX.w * iw),
      h: Math.floor(HARDCODED_BOX.h * ih),
    });

    ctx.strokeStyle = '#E2823A';
    ctx.lineWidth = 1.5;
    const corner = Math.min(22, bw * 0.2, bh * 0.2);
    const c2 = (x, y, dx1, dy1, dx2, dy2) => {
      ctx.beginPath();
      ctx.moveTo(x + dx1, y + dy1);
      ctx.lineTo(x, y);
      ctx.lineTo(x + dx2, y + dy2);
      ctx.stroke();
    };
    c2(bx, by, 0, corner, corner, 0);
    c2(bx + bw, by, -corner, 0, 0, corner);
    c2(bx, by + bh, 0, -corner, corner, 0);
    c2(bx + bw, by + bh, -corner, 0, 0, -corner);

    ctx.setLineDash([4, 4]);
    ctx.lineWidth = 0.8;
    ctx.strokeStyle = 'rgba(226, 130, 58, 0.35)';
    ctx.strokeRect(bx, by, bw, bh);
    ctx.setLineDash([]);

    ctx.font = '600 9px "IBM Plex Mono", monospace';
    ctx.fillStyle = '#E2823A';
    ctx.fillText('CROP REGION // HARDCODED', bx, by - 8);
  }, [camera.active, camera.videoRef]);

  useEffect(() => {
    if (!camera.active) return;
    let raf;
    const loop = () => { drawOverlay(); raf = requestAnimationFrame(loop); };
    loop();
    return () => cancelAnimationFrame(raf);
  }, [camera.active, drawOverlay]);

  // Capture helpers 
  const getFullFrame = useCallback(async () => {
    const v = camera.videoRef.current;
    if (!v) throw new Error('Camera not ready');
    return frameToBlob(v);
  }, [camera.videoRef]);

  const getCrop = useCallback(async () => {
    const v = camera.videoRef.current;
    if (!v) throw new Error('Camera not ready');
    const { blob, dataUrl, pixelBox: pb } = await cropToBlob(v);
    setLastCropPreview({ dataUrl, pixelBox: pb });
    return blob;
  }, [camera.videoRef]);

  // Verify expects a 128x128 input — resize the crop to match.
  // Without keypoint-based alignment the similarity score isn't meaningful,
  // but this satisfies the backend's dimension check so the endpoint
  // returns a response instead of 400.
  const get128Crop = useCallback(async () => {
    const v = camera.videoRef.current;
    if (!v) throw new Error('Camera not ready');
    const { dataUrl, pixelBox: pb } = await cropToBlob(v);
    setLastCropPreview({ dataUrl, pixelBox: pb });

    const img = await new Promise((resolve, reject) => {
      const i = new Image();
      i.onload = () => resolve(i);
      i.onerror = reject;
      i.src = dataUrl;
    });
    const c = document.createElement('canvas');
    c.width = 128;
    c.height = 128;
    c.getContext('2d').drawImage(img, 0, 0, 128, 128);
    return new Promise((res) => c.toBlob(res, 'image/jpeg', 0.9));
  }, [camera.videoRef]);

  const requireCam = () => {
    if (!camera.active) {
      return { ok: false, status: 0, latency: 0, error: 'Start the camera first.' };
    }
    return null;
  };

  // Endpoint runners 
  const runPipeline = async () => {
    const e = requireCam(); if (e) return e;
    return postPipeline(await getFullFrame());
  };
  const runDetect = async () => {
    const e = requireCam(); if (e) return e;
    return postDetect(await getFullFrame());
  };
  const runEmotion = async () => {
    const e = requireCam(); if (e) return e;
    return postEmotion(await getCrop());
  };
  const runSpoof = async () => {
    const e = requireCam(); if (e) return e;
    return postSpoof(await getCrop());
  };
  // Verify expects pre-aligned 128x128 — see get128Crop comment above.
  const runVerify = async () => {
    const e = requireCam(); if (e) return e;
    return postVerify(await get128Crop());
  };
  // Register sends the FULL FRAME — the backend's inference_service.register_inference()
  // runs detection + alignment internally. Sending a pre-crop breaks that flow.
  const runRegister = async (inputs) => {
    const e = requireCam(); if (e) return e;
    const res = await postRegister(await getFullFrame(), inputs.person_id, inputs.person_name);
    if (res.ok) {
      onRegistryChange?.();
      getVerificationStatus().then(setVerificationStatus);
    }
    return res;
  };

  return (
    <div className="diag">
      {/* ── LEFT: Camera with hardcoded box ── */}
      <div className="diag__cam-col">
        <div className="camera">
          <div className="camera__head">
            <div className="camera__head-title">
              <span className="camera__section-num">§ DIAG · 01</span>
              <span className="camera__section-title">Capture Apparatus</span>
            </div>
            <span className="t-label">Hardcoded crop region</span>
          </div>

          <div className="camera__viewport" ref={containerRef} style={{ minHeight: 360 }}>
            <span className="camera__corner tr" />
            <span className="camera__corner bl" />

            <video
              ref={camera.videoRef}
              className="camera__video"
              autoPlay playsInline muted
              style={{ display: camera.active ? 'block' : 'none' }}
            />
            <canvas ref={overlayRef} className="camera__overlay" />

            {camera.active && <div className="camera__rec">REC · DIAG</div>}

            {!camera.active && (
              <div className="camera__off">
                <CameraOffIcon className="camera__off-icon" />
                <div>
                  <div className="camera__off-title">Camera dormant</div>
                  <div className="camera__off-sub">Activate to begin testing</div>
                </div>
              </div>
            )}
          </div>

          <div className="camera__controls">
            <div className="camera__controls-buttons">
              <button
                className={`btn ${camera.active ? 'btn--ghost' : 'btn--primary'}`}
                onClick={() => camera.active ? camera.stop() : camera.start()}
              >
                {camera.active ? <PauseIcon /> : <PlayIcon />}
                {camera.active ? 'Halt camera' : 'Start camera'}
              </button>
            </div>
            <div className="camera__counter">
              <strong className="t-num">{camera.active ? 'LIVE' : 'OFF'}</strong>
              &nbsp;&middot;&nbsp; hardcoded {(HARDCODED_BOX.w * 100).toFixed(0)}%×{(HARDCODED_BOX.h * 100).toFixed(0)}% region
            </div>
          </div>
        </div>

        {/* Box dimensions strip */}
        <div className="diag__box-info">
          <div className="diag__box-info-cell">
            <div className="diag__box-info-label">Box X</div>
            <div className="diag__box-info-value">{pixelBox ? `${pixelBox.x}px` : '—'}</div>
          </div>
          <div className="diag__box-info-cell">
            <div className="diag__box-info-label">Box Y</div>
            <div className="diag__box-info-value">{pixelBox ? `${pixelBox.y}px` : '—'}</div>
          </div>
          <div className="diag__box-info-cell">
            <div className="diag__box-info-label">Width</div>
            <div className="diag__box-info-value">{pixelBox ? `${pixelBox.w}px` : '—'}</div>
          </div>
          <div className="diag__box-info-cell">
            <div className="diag__box-info-label">Height</div>
            <div className="diag__box-info-value">{pixelBox ? `${pixelBox.h}px` : '—'}</div>
          </div>
        </div>

        {/* Crop preview */}
        {lastCropPreview && (
          <div className="crop-preview">
            <img className="crop-preview__img" src={lastCropPreview.dataUrl} alt="Last crop sent" />
            <div className="crop-preview__meta">
              <div className="crop-preview__label">Last crop sent</div>
              <div className="crop-preview__value">
                {lastCropPreview.pixelBox.w}×{lastCropPreview.pixelBox.h}px
              </div>
            </div>
          </div>
        )}
      </div>

      {/* ── RIGHT: Status + Endpoint test cards ── */}
      <div className="diag__test-col">

        {/* Backend connection */}
        <div className={`conn-panel ${backendReachable === true ? 'ok' : backendReachable === false ? 'err' : ''}`}>
          <div>
            <div className="conn-panel__label">Backend</div>
            <div className={`conn-panel__value ${backendReachable === true ? 'ok' : backendReachable === false ? 'err' : ''}`}>
              {backendReachable === null ? 'Checking…' : backendReachable ? 'Reachable' : 'Unreachable'}
            </div>
            <div className="conn-panel__url">
              proxied through /api · check VITE_BACKEND_URL in .env if offline
            </div>
          </div>
          <div className={`signal-dot ${backendReachable === true ? 'go' : backendReachable === false ? 'stop' : ''}`} />
        </div>

        {/* Qdrant / verification status */}
        {verificationStatus && (
          <div className="conn-panel ok">
            <div>
              <div className="conn-panel__label">Vector Store · Qdrant</div>
              <div className="conn-panel__value ok">
                {verificationStatus.collection} · {verificationStatus.points} embeddings
              </div>
              <div className="conn-panel__url">
                vector size {verificationStatus.vector_size} · top_k {verificationStatus.top_k}
                {' · '}max {verificationStatus.max_registration_images} images/register
              </div>
            </div>
            <div className="signal-dot go" />
          </div>
        )}

        <div className="diag__intro">
          <div className="diag__intro-title">How to use this page</div>
          <div className="diag__intro-text">
            Start the camera, then press <em>Run test</em> on any card to send a
            request to that endpoint. Green border = success, red = error. Some
            cards send the full frame (the backend does its own face detection),
            others send only the hardcoded crop region.
          </div>
        </div>

        <EndpointCard
          title="Frame Pipeline"
          path={PATHS.pipeline}
          description="Full one-shot analysis. Sends the entire frame and expects detection + recognition + emotion + anti-spoofing back."
          onRun={runPipeline}
          hint="Sends full frame"
          summaryRenderer={(b) => [
            { key: 'Image', val: `${b.image_width}×${b.image_height}` },
            { key: 'Faces', val: b.faces?.length ?? 0, kind: (b.faces?.length > 0) ? 'go' : '' },
            ...(b.faces?.[0] ? [
              { key: 'Emotion',  val: b.faces[0].emotion?.label || '—' },
              { key: 'Liveness', val: b.faces[0].anti_spoofing?.label || '—',
                kind: b.faces[0].anti_spoofing?.label === 'real' ? 'go'
                   : b.faces[0].anti_spoofing?.label === 'spoof' ? 'stop' : '' },
              { key: 'Identity', val: b.faces[0].recognition?.label || '—',
                kind: b.faces[0].recognition?.matched ? 'go' : '' },
            ] : []),
          ]}
        />

        <EndpointCard
          title="Face Detection"
          path={PATHS.detect}
          description="Returns bounding boxes and 5-point landmarks for any faces found in the frame."
          onRun={runDetect}
          hint="Sends full frame"
          summaryRenderer={(b) => [
            { key: 'Image', val: `${b.image_width}×${b.image_height}` },
            { key: 'Faces', val: b.faces?.length ?? 0, kind: (b.faces?.length > 0) ? 'go' : '' },
            ...(b.faces?.[0] ? [
              { key: 'First bbox',
                val: `${b.faces[0].bbox.x.toFixed(2)}, ${b.faces[0].bbox.y.toFixed(2)}, ${b.faces[0].bbox.w.toFixed(2)}, ${b.faces[0].bbox.h.toFixed(2)}` },
              { key: 'Keypoints', val: b.faces[0].keypoints?.length ?? 0 },
              { key: 'Confidence', val: `${(b.faces[0].detection_confidence * 100).toFixed(1)}%`, kind: 'accent' },
            ] : []),
          ]}
        />

        <EndpointCard
          title="Emotion"
          path={PATHS.emotion}
          description="Classifies the cropped face into an emotion label. Sends only the hardcoded crop region."
          onRun={runEmotion}
          hint="Sends crop only"
          summaryRenderer={(b) => [
            { key: 'Label',      val: b.emotion?.label || '—', kind: 'accent' },
            { key: 'Confidence', val: `${((b.emotion?.confidence ?? 0) * 100).toFixed(1)}%` },
          ]}
        />

        <EndpointCard
          title="Anti-Spoofing"
          path={PATHS.spoof}
          description="Decides if the cropped face is live or spoof (printed photo, screen replay). Sends only the crop."
          onRun={runSpoof}
          hint="Sends crop only"
          summaryRenderer={(b) => [
            { key: 'Label',
              val: b.anti_spoofing?.label || '—',
              kind: b.anti_spoofing?.label === 'real' ? 'go'
                 : b.anti_spoofing?.label === 'spoof' ? 'stop' : '' },
            { key: 'Confidence', val: `${((b.anti_spoofing?.confidence ?? 0) * 100).toFixed(1)}%` },
          ]}
        />

        <EndpointCard
          title="Recognition (Verify)"
          path={PATHS.verify}
          description="Sends a 128×128 resized crop to match the backend's expected input format. Note: the backend's /verify route doesn't run detection + face alignment internally, so the similarity score is unreliable as a standalone test — use the Pipeline card for meaningful end-to-end verification."
          onRun={runVerify}
          hint="Sends 128×128 crop"
          summaryRenderer={(b) => [
            { key: 'Label',      val: b.recognition?.label || '—', kind: 'accent' },
            { key: 'Matched',    val: b.recognition?.matched ? 'YES' : 'no',
              kind: b.recognition?.matched ? 'go' : '' },
            { key: 'Confidence', val: `${((b.recognition?.confidence ?? 0) * 100).toFixed(1)}%` },
          ]}
        />

        <EndpointCard
          title="Register Identity"
          path={PATHS.register}
          description="Stores a face embedding in Qdrant under the given ID. Sends the full frame — the backend runs detection + alignment internally before producing the embedding."
          inputs={[
            { key: 'person_id',   label: 'person_id',   placeholder: 'e.g. emp_021', required: true },
            { key: 'person_name', label: 'person_name', placeholder: 'optional display name' },
          ]}
          onRun={runRegister}
          hint="Sends full frame"
          summaryRenderer={(b) => [
            { key: 'Person ID', val: b.person_id || '—', kind: 'accent' },
            { key: 'Status',    val: b.status || '—' },
            { key: 'Message',   val: b.message || '' },
          ]}
        />
      </div>
    </div>
  );
}