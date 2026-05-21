import React, { useEffect, useState } from 'react';

export default function TopBar({ cameraActive, connected, route, onNavigate }) {
  const [now, setNow] = useState(new Date());
  useEffect(() => {
    const t = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(t);
  }, []);

  const dateStr = now.toLocaleDateString('en-AU', {
    weekday: 'long', day: 'numeric', month: 'long', year: 'numeric',
  }).toUpperCase();
  const timeStr = now.toLocaleTimeString('en-AU', { hour12: false });

  let connText = 'Standby';
  let connClass = '';
  if (connected === true) { connText = 'Online'; connClass = 'go'; }
  else if (connected === false) { connText = 'Offline'; connClass = 'stop'; }

  return (
    <header className="topbar">
      <div className="topbar__brand">
        <div className="topbar__brand-mark">Facial Recognition <em>with Emotion and Liveness</em></div>
      </div>

      <nav className="topbar__nav">
        <button
          className={`topbar__nav-item ${route === 'main' ? 'active' : ''}`}
          onClick={() => onNavigate('main')}
        >
          § 01 <em>Main Page</em>
        </button>
        <button
          className={`topbar__nav-item ${route === 'diagnostics' ? 'active' : ''}`}
          onClick={() => onNavigate('diagnostics')}
        >
          § 02 <em>Dev Page</em>
        </button>
      </nav>

      <div className="topbar__meta">
        <span className="topbar__meta-item">
          <span className={`signal-dot ${connClass}`} />
          {connText}
        </span>
        <span className="topbar__meta-item">
          <span className={`signal-dot ${cameraActive ? 'go' : ''}`} />
          {cameraActive ? 'Camera Live' : 'Camera Idle'}
        </span>
        <span className="topbar__meta-item t-num">
          {dateStr} &middot; {timeStr}
        </span>
      </div>
    </header>
  );
}
