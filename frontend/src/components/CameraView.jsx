import React, { useEffect, useRef, useState } from 'react';
import { drawFaces } from '../utils/drawing.js';
import { PlayIcon, PauseIcon, PlusIcon, CameraOffIcon } from './Icons.jsx';

export default function CameraView({
  videoRef,
  active,
  analysis,
  checkinCount,
  onToggle,
  onRegister,
}) {
  const overlayRef = useRef(null);
  const containerRef = useRef(null);
  const [timecode, setTimecode] = useState('');

  // Resize canvas to match container
  useEffect(() => {
    const resize = () => {
      const overlay = overlayRef.current;
      const container = containerRef.current;
      if (!overlay || !container) return;
      const rect = container.getBoundingClientRect();
      overlay.width = rect.width;
      overlay.height = rect.height;
    };
    resize();
    window.addEventListener('resize', resize);
    return () => window.removeEventListener('resize', resize);
  }, [active]);

  // Draw boxes when analysis arrives
  useEffect(() => {
    const ctx = overlayRef.current?.getContext('2d');
    if (!ctx) return;
    if (!analysis) {
      ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
      return;
    }
    drawFaces(ctx, analysis.faces, analysis.image_width, analysis.image_height, true);
  }, [analysis]);

  // Timecode ticker (visual flourish)
  useEffect(() => {
    if (!active) return;
    const t = setInterval(() => {
      const d = new Date();
      const pad = (n, w = 2) => String(n).padStart(w, '0');
      setTimecode(
        `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}.${pad(Math.floor(d.getMilliseconds() / 10))}`
      );
    }, 50);
    return () => clearInterval(t);
  }, [active]);

  return (
    <div className="camera">
      <div className="camera__head">
        <div className="camera__head-title">
          <span className="camera__section-num">§ 01</span>
          <span className="camera__section-title">Field of View</span>
        </div>
        <span className="t-label">Live, real-time capture</span>
      </div>

      <div className="camera__viewport" ref={containerRef}>
        <span className="camera__corner tr" />
        <span className="camera__corner bl" />

        <video
          ref={videoRef}
          className="camera__video"
          autoPlay
          playsInline
          muted
          style={{ display: active ? 'block' : 'none' }}
        />
        <canvas ref={overlayRef} className="camera__overlay" />

        {active && <div className="camera__rec">REC &middot; 01</div>}
        {active && timecode && <div className="camera__timecode">{timecode}</div>}

        {!active && (
          <div className="camera__off">
            <CameraOffIcon className="camera__off-icon" />
            <div>
              <div className="camera__off-title">Camera dormant</div>
              <div className="camera__off-sub">Press <em>Turn on Camera</em> to activate</div>
            </div>
          </div>
        )}
      </div>

      <div className="camera__controls">
        <div className="camera__controls-buttons">
          <button
            className={`btn ${active ? 'btn--ghost' : 'btn--primary'}`}
            onClick={onToggle}
          >
            {active ? <PauseIcon /> : <PlayIcon />}
            {active ? 'Turn off camera' : 'Turn on Camera'}
          </button>
          <button
            className="btn btn--accent"
            onClick={onRegister}
            disabled={!active}
          >
            <PlusIcon />
            Register Face
          </button>
        </div>
        <div className="camera__counter">
          <strong className="t-num">{String(checkinCount).padStart(2, '0')}</strong>
          checked in &middot; today
        </div>
      </div>
    </div>
  );
}
