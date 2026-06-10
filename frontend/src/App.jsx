import React, { useCallback, useEffect, useState } from 'react';
import TopBar from './components/TopBar.jsx';
import CameraView from './components/CameraView.jsx';
import DetectionPanel from './components/DetectionPanel.jsx';
import RegisteredFaces from './components/RegisteredFaces.jsx';
import AttendanceLog from './components/AttendanceLog.jsx';
import Toasts from './components/Toasts.jsx';
import RegistrationView from './components/RegistrationView.jsx';
import VideoUploadView from './components/VideoUploadView.jsx';
import { useCamera } from './hooks/useCamera.js';
import { useFrameAnalysis } from './hooks/useFrameAnalysis.js';
import { useToasts } from './hooks/useToasts.js';
import { ping, getVerificationStatus } from './services/api.js';

const DEDUP_WINDOW_MS = 10_000;

export default function App() {
  // Routing 
  const routeFromHash = useCallback(() => {
    if (window.location.hash === '#register') return 'register';
    if (window.location.hash === '#video') return 'video';
    return 'main';
  }, []);
  const [route, setRoute] = useState(routeFromHash);
  const navigate = useCallback((r) => {
    setRoute(r);
    window.location.hash = r === 'main' ? '' : r;
  }, []);
  useEffect(() => {
    const onHashChange = () => {
      setRoute(routeFromHash());
    };
    window.addEventListener('hashchange', onHashChange);
    return () => window.removeEventListener('hashchange', onHashChange);
  }, [routeFromHash]);

  // Shared camera
  const camera = useCamera();

  // Toasts
  const { toasts, push } = useToasts();

  // Main-page state
  const { analysis, connected } = useFrameAnalysis(
    camera.captureBlob,
    camera.active && route === 'main',
  );

  // Registry — local mirror plus backend-reported total count
  const [registry, setRegistry] = useState({});
  const [backendRegistryCount, setBackendRegistryCount] = useState(null);
  const [selectedName, setSelectedName] = useState(null);

  const [log, setLog] = useState([]);
  const [seenToday] = useState(() => new Set());
  const [pingResult, setPingResult] = useState(null);

  // Helper: refresh registered count from backend
  const refreshBackendCount = useCallback(async () => {
    const status = await getVerificationStatus();
    if (status && typeof status.points === 'number') {
      setBackendRegistryCount(status.points);
    }
  }, []);

  // On mount: ping + backend count
  useEffect(() => {
    ping().then((ok) => {
      setPingResult(ok);
      if (ok) push('Backend reachable', 'success');
      else push('Backend not reachable — start it to use the application', 'error', 6000);
    });
    refreshBackendCount();
  }, [push, refreshBackendCount]);

  // Camera errors
  useEffect(() => {
    if (camera.error) push(`Camera: ${camera.error}`, 'error');
  }, [camera.error, push]);

  // A route owns the camera while it is visible. Stop it before moving
  // between the attendance and registration camera elements.
  useEffect(() => {
    camera.stop();
  }, [route]);

  // Attendance ingestion (main page only)
  useEffect(() => {
    if (route !== 'main') return;
    const face = analysis?.faces?.[0];
    if (!face) return;
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

  // Handlers
  const handleToggleCamera = useCallback(() => {
    if (camera.active) camera.stop();
    else camera.start();
  }, [camera]);

  const handleRemove = useCallback(() => {
    if (!selectedName) return;
    if (!confirm(`Remove "${selectedName}" from your local list? (Backend record remains until a delete endpoint is added.)`)) return;
    setRegistry((r) => { const n = { ...r }; delete n[selectedName]; return n; });
    setSelectedName(null);
    push('Removed from local list', 'success');
  }, [selectedName, push]);

  // Effective connection status for topbar
  const effectiveConnected = route === 'main' ? connected : pingResult;

  return (
    <div className="app">
      <TopBar
        cameraActive={camera.active}
        connected={effectiveConnected}
        route={route}
        onNavigate={navigate}
      />

      {route === 'register' ? (
        <RegistrationView
          camera={camera}
          push={push}
          onRegistered={refreshBackendCount}
        />
      ) : route === 'video' ? (
        <VideoUploadView push={push} />
      ) : (
        <main className="app__main">
          <CameraView
            videoRef={camera.videoRef}
            active={camera.active}
            analysis={analysis}
            checkinCount={seenToday.size}
            onToggle={handleToggleCamera}
          />
          <aside className="column">
            <DetectionPanel analysis={analysis} />
            <RegisteredFaces
              entries={registry}
              selectedName={selectedName}
              backendCount={backendRegistryCount}
              onSelect={(n) => setSelectedName(n === selectedName ? null : n)}
              onRemove={handleRemove}
              onRefresh={refreshBackendCount}
            />
            <AttendanceLog entries={log} />
          </aside>
        </main>
      )}

      <Toasts toasts={toasts} />
    </div>
  );
}
