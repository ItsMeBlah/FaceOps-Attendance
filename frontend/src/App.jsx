import React, { useCallback, useEffect, useState } from 'react';
import TopBar from './components/TopBar.jsx';
import CameraView from './components/CameraView.jsx';
import DetectionPanel from './components/DetectionPanel.jsx';
import RegisteredFaces from './components/RegisteredFaces.jsx';
import AttendanceLog from './components/AttendanceLog.jsx';
import RegisterModal from './components/RegisterModal.jsx';
import Toasts from './components/Toasts.jsx';
import DiagnosticsView from './components/DiagnosticsView.jsx';
import { useCamera } from './hooks/useCamera.js';
import { useFrameAnalysis } from './hooks/useFrameAnalysis.js';
import { useToasts } from './hooks/useToasts.js';
import { registerFace, ping } from './services/api.js';

const DEDUP_WINDOW_MS = 10_000;

export default function App() {
  // ── Routing ──
  // Simple hash routing — survives reloads and is shareable
  const [route, setRoute] = useState(() =>
    window.location.hash === '#diagnostics' ? 'diagnostics' : 'main'
  );
  const navigate = useCallback((r) => {
    setRoute(r);
    window.location.hash = r === 'diagnostics' ? 'diagnostics' : '';
  }, []);
  useEffect(() => {
    const onHashChange = () => {
      setRoute(window.location.hash === '#diagnostics' ? 'diagnostics' : 'main');
    };
    window.addEventListener('hashchange', onHashChange);
    return () => window.removeEventListener('hashchange', onHashChange);
  }, []);

  // ── Shared camera (single source — same instance used by both pages) ──
  const camera = useCamera();

  // ── Toasts ──
  const { toasts, push } = useToasts();

  // ── Main-page state ──
  const { analysis, connected } = useFrameAnalysis(
    camera.captureBlob,
    camera.active && route === 'main',
  );
  const [registry, setRegistry] = useState({});
  const [selectedName, setSelectedName] = useState(null);
  const [log, setLog] = useState([]);
  const [seenToday] = useState(() => new Set());
  const [modalOpen, setModalOpen] = useState(false);
  const [pingResult, setPingResult] = useState(null);

  // ── Backend ping on first mount ──
  useEffect(() => {
    ping().then((ok) => {
      setPingResult(ok);
      if (ok) push('Backend reachable', 'success');
      else push('Backend not reachable — start it before running tests', 'error', 6000);
    });
  }, [push]);

  // ── Camera errors ──
  useEffect(() => {
    if (camera.error) push(`Camera: ${camera.error}`, 'error');
  }, [camera.error, push]);

  // ── Attendance ingestion (main page only) ──
  useEffect(() => {
    if (route !== 'main') return;
    const face = analysis?.faces?.[0];
    if (!face) return;
    if (analysis._placeholder) return;
    const rec = face.recognition;
    if (!rec?.matched || rec.label === 'unknown') return;

    const name = rec.label;
    const isLive = face.anti_spoofing?.label === 'real';
    const emotionLabel = face.emotion?.label || '—';

    setLog((prev) => {
      const last = prev.find((e) => e.name === name);
      if (last && Date.now() - last._ts < DEDUP_WINDOW_MS) return prev;
      const isNew = !seenToday.has(name);
      seenToday.add(name);
      const now = new Date();
      const time = now.toLocaleTimeString('en-AU', { hour12: false });
      setRegistry((r) => ({
        ...r,
        [name]: {
          ...(r[name] || {}),
          lastSeen: now.toLocaleString('en-AU', {
            day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit', hour12: false,
          }),
        },
      }));
      return [{
        id: `${name}-${now.getTime()}`,
        name, time, emotion: emotionLabel, isLive, isNew, _ts: now.getTime(),
      }, ...prev].slice(0, 50);
    });
  }, [analysis, route, seenToday]);

  // ── Handlers ──
  const handleToggleCamera = useCallback(() => {
    if (camera.active) camera.stop();
    else camera.start();
  }, [camera]);

  const handleOpenRegister = useCallback(() => {
    if (!camera.active) { push('Begin Watch first to enrol someone', 'warn'); return; }
    setModalOpen(true);
  }, [camera.active, push]);

  const handleRegister = useCallback(async (name) => {
    const blob = await camera.captureBlob();
    if (!blob) throw new Error('Could not capture a frame');
    try { await registerFace(blob, name); }
    catch (e) { throw new Error(`Registration failed: ${e.message}`); }
    setRegistry((r) => ({
      ...r,
      [name]: { registeredAt: new Date().toISOString(), lastSeen: null },
    }));
    push(`${name} enrolled`, 'success');
    setModalOpen(false);
  }, [camera, push]);

  const handleRemove = useCallback(() => {
    if (!selectedName) return;
    if (!confirm(`Strike "${selectedName}" from the registry?`)) return;
    setRegistry((r) => { const n = { ...r }; delete n[selectedName]; return n; });
    setSelectedName(null);
    push('Entry removed from local registry', 'success');
  }, [selectedName, push]);

  // Effective connection status used by the topbar
  const effectiveConnected = route === 'main' ? connected : pingResult;

  return (
    <div className="app">
      <TopBar
        cameraActive={camera.active}
        connected={effectiveConnected}
        route={route}
        onNavigate={navigate}
      />

      {route === 'diagnostics' ? (
        <DiagnosticsView camera={camera} />
      ) : (
        <main className="app__main">
          <CameraView
            videoRef={camera.videoRef}
            active={camera.active}
            analysis={analysis}
            checkinCount={seenToday.size}
            onToggle={handleToggleCamera}
            onRegister={handleOpenRegister}
          />
          <aside className="column">
            <DetectionPanel analysis={analysis} />
            <RegisteredFaces
              entries={registry}
              selectedName={selectedName}
              onSelect={(n) => setSelectedName(n === selectedName ? null : n)}
              onRemove={handleRemove}
            />
            <AttendanceLog entries={log} />
          </aside>
        </main>
      )}

      <RegisterModal
        open={modalOpen}
        onClose={() => setModalOpen(false)}
        onConfirm={handleRegister}
      />

      <Toasts toasts={toasts} />
    </div>
  );
}
