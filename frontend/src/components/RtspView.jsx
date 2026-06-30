import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Link, Loader2, Play, Radio, Square, Trash2, Video } from 'lucide-react';
import {
  getRtspFeedUrl,
  getRtspStreamStatus,
  startRtspStream,
  stopRtspStream,
} from '../services/api.js';

const STORAGE_KEY = 'faceops.rtsp.streams.v2';

function createStream(url, name, cameraId, status) {
  const createdAt = new Date().toISOString();
  return {
    id: cameraId || `${Date.now()}-${Math.random().toString(16).slice(2)}`,
    name: name.trim() || defaultCameraName(url),
    url: url.trim(),
    cameraId,
    createdAt,
    status: status?.status || 'starting',
    running: Boolean(status?.running),
    error: status?.error || '',
    frameCount: status?.frame_count || 0,
    publishedCount: status?.published_count || 0,
    lastFrameAt: status?.last_frame_at || '',
    feedVersion: Date.now(),
  };
}

function defaultCameraName(url) {
  try {
    const parsed = new URL(url);
    return parsed.hostname ? `Camera ${parsed.hostname}` : 'RTSP Camera';
  } catch {
    return 'RTSP Camera';
  }
}

function loadStreams() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    const parsed = raw ? JSON.parse(raw) : [];
    if (!Array.isArray(parsed)) return [];
    return parsed.map((stream) => ({
      ...stream,
      status: stream.status || 'stopped',
      running: Boolean(stream.running),
      error: stream.error || '',
      frameCount: stream.frameCount || 0,
      publishedCount: stream.publishedCount || 0,
      feedVersion: stream.feedVersion || Date.now(),
    }));
  } catch {
    return [];
  }
}

function isValidRtspUrl(value) {
  try {
    const parsed = new URL(value);
    return parsed.protocol === 'rtsp:';
  } catch {
    return false;
  }
}

function formatCreatedAt(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return '--';
  return date.toLocaleString('en-AU', {
    day: '2-digit',
    month: 'short',
    hour: '2-digit',
    minute: '2-digit',
    hour12: false,
  });
}

function statusLabel(stream) {
  if (!stream) return 'Idle';
  if (stream.status === 'error') return 'Error';
  if (stream.status === 'starting') return 'Starting';
  if (stream.running) return 'Live';
  return 'Stopped';
}

function statusClass(stream) {
  if (!stream) return '';
  if (stream.status === 'error') return 'stop';
  if (stream.status === 'starting') return 'warn';
  if (stream.running) return 'go';
  return '';
}

export default function RtspView({ push }) {
  const [streams, setStreams] = useState(loadStreams);
  const [url, setUrl] = useState('');
  const [name, setName] = useState('');
  const [cameraId, setCameraId] = useState('');
  const [selectedId, setSelectedId] = useState(null);
  const [busyId, setBusyId] = useState(null);
  const [formBusy, setFormBusy] = useState(false);
  const urlInputRef = useRef(null);

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(streams));
  }, [streams]);

  useEffect(() => {
    if (!selectedId && streams.length > 0) {
      setSelectedId(streams[0].id);
    }
    if (selectedId && !streams.some((stream) => stream.id === selectedId)) {
      setSelectedId(streams[0]?.id || null);
    }
  }, [selectedId, streams]);

  const selectedStream = useMemo(
    () => streams.find((stream) => stream.id === selectedId) || streams[0] || null,
    [selectedId, streams],
  );

  const updateStream = useCallback((streamId, patch) => {
    setStreams((current) => current.map((stream) => (
      stream.id === streamId ? { ...stream, ...patch } : stream
    )));
  }, []);

  const applyStatus = useCallback((streamId, status) => {
    setStreams((current) => current.map((stream) => {
      if (stream.id !== streamId) return stream;
      return {
        ...stream,
        cameraId: status.camera_id,
        status: status.status,
        running: status.running,
        error: status.error || '',
        frameCount: status.frame_count || 0,
        publishedCount: status.published_count || 0,
        lastFrameAt: status.last_frame_at || '',
        feedVersion: status.running && !stream.running ? Date.now() : stream.feedVersion,
      };
    }));
  }, []);

  const startStream = useCallback(async (stream) => {
    if (!stream || busyId === stream.id) return;
    setBusyId(stream.id);
    updateStream(stream.id, { status: 'starting', error: '' });
    try {
      const status = await startRtspStream({
        rtsp_url: stream.url,
        camera_id: stream.cameraId || undefined,
      });
      applyStatus(stream.id, status);
      push?.('RTSP stream started', 'success');
    } catch (error) {
      updateStream(stream.id, {
        status: 'error',
        running: false,
        error: error.message || 'Unable to start RTSP stream',
      });
      push?.(error.message || 'Unable to start RTSP stream', 'error');
    } finally {
      setBusyId(null);
    }
  }, [applyStatus, busyId, push, updateStream]);

  const stopStream = useCallback(async (stream) => {
    if (!stream?.cameraId || busyId === stream.id) return;
    setBusyId(stream.id);
    try {
      const status = await stopRtspStream(stream.cameraId);
      applyStatus(stream.id, status);
      push?.('RTSP stream stopped', 'success');
    } catch (error) {
      updateStream(stream.id, {
        status: 'error',
        running: false,
        error: error.message || 'Unable to stop RTSP stream',
      });
      push?.(error.message || 'Unable to stop RTSP stream', 'error');
    } finally {
      setBusyId(null);
    }
  }, [applyStatus, busyId, push, updateStream]);

  const addAndStartStream = useCallback(async (event) => {
    event.preventDefault();
    const trimmedUrl = url.trim();

    if (!isValidRtspUrl(trimmedUrl)) {
      push?.('Enter a valid rtsp:// stream URL', 'error');
      urlInputRef.current?.focus();
      return;
    }

    const requestedCameraId = cameraId.trim();
    if (requestedCameraId && streams.some((stream) => stream.cameraId === requestedCameraId)) {
      push?.('This camera ID is already in the stream list', 'warn');
      return;
    }

    setFormBusy(true);
    try {
      const status = await startRtspStream({
        rtsp_url: trimmedUrl,
        camera_id: requestedCameraId || undefined,
      });
      const stream = createStream(trimmedUrl, name, status.camera_id, status);
      setStreams((current) => [stream, ...current]);
      setSelectedId(stream.id);
      setUrl('');
      setName('');
      setCameraId('');
      push?.('RTSP stream started', 'success');
    } catch (error) {
      push?.(error.message || 'Unable to start RTSP stream', 'error');
    } finally {
      setFormBusy(false);
    }
  }, [cameraId, name, push, streams, url]);

  const removeStream = useCallback(async (streamId) => {
    const stream = streams.find((item) => item.id === streamId);
    if (!stream) return;
    if (!confirm(`Delete "${stream.name}" from this RTSP page?`)) return;

    if (stream.running && stream.cameraId) {
      try {
        await stopRtspStream(stream.cameraId);
      } catch {
        push?.('Stream removed locally, but backend stop failed', 'warn');
      }
    }

    setStreams((current) => current.filter((item) => item.id !== streamId));
    push?.('RTSP stream removed', 'success');
  }, [push, streams]);

  useEffect(() => {
    const activeStreams = streams.filter((stream) => (
      stream.cameraId && (stream.running || stream.status === 'starting')
    ));
    if (activeStreams.length === 0) return undefined;

    const timer = setInterval(() => {
      activeStreams.forEach((stream) => {
        getRtspStreamStatus(stream.cameraId)
          .then((status) => applyStatus(stream.id, status))
          .catch((error) => {
            updateStream(stream.id, {
              status: 'error',
              running: false,
              error: error.message || 'Unable to read RTSP status',
            });
          });
      });
    }, 2000);

    return () => clearInterval(timer);
  }, [applyStatus, streams, updateStream]);

  return (
    <main className="rtsp">
      <section className="rtsp__stage-col">
        <header className="camera__head">
          <div className="camera__head-title">
            <span className="camera__section-num">§ 01</span>
            <span className="camera__section-title">RTSP Monitor</span>
          </div>
          <span className="t-label">{streams.length} stream{streams.length === 1 ? '' : 's'} configured</span>
        </header>

        <StreamGrid
          busyId={busyId}
          onRemove={removeStream}
          onSelect={setSelectedId}
          onStart={startStream}
          onStop={stopStream}
          selectedId={selectedStream?.id}
          streams={streams}
        />

        <form className="rtsp__form" onSubmit={addAndStartStream}>
          <label className="rtsp__field rtsp__field--name">
            <span>Name</span>
            <input
              value={name}
              onChange={(event) => setName(event.target.value)}
              placeholder="Front door"
            />
          </label>
          <label className="rtsp__field rtsp__field--camera">
            <span>Camera ID</span>
            <input
              value={cameraId}
              onChange={(event) => setCameraId(event.target.value)}
              placeholder="front-door"
            />
          </label>
          <label className="rtsp__field">
            <span>RTSP URL</span>
            <input
              ref={urlInputRef}
              value={url}
              onChange={(event) => setUrl(event.target.value)}
              placeholder="rtsp://username:password@192.168.1.10:554/stream1"
            />
          </label>
          <button className="btn btn--primary" disabled={formBusy} type="submit">
            {formBusy ? <Loader2 /> : <Play />}
            Start stream
          </button>
        </form>
      </section>

      <aside className="column rtsp__aside">
        <section className="section">
          <header className="section__head">
            <div className="section__head-title">
              <span className="section__num">§ 02</span>
              <span className="section__title">Stream List</span>
            </div>
            <span className="section__aside">Backend sessions</span>
          </header>

          <div className="rtsp__list">
            {streams.length === 0 ? (
              <div className="rtsp__empty">
                <Video />
                <span>No RTSP streams added</span>
              </div>
            ) : (
              streams.map((stream, index) => (
                <button
                  className={`rtsp__list-item ${stream.id === selectedStream?.id ? 'active' : ''}`}
                  key={stream.id}
                  onClick={() => setSelectedId(stream.id)}
                  type="button"
                >
                  <span className="rtsp__list-index">{String(index + 1).padStart(2, '0')}</span>
                  <span className="rtsp__list-main">
                    <strong>{stream.name}</strong>
                    <em>{stream.cameraId || stream.url}</em>
                  </span>
                  <span className={`rtsp__list-state ${statusClass(stream)}`}>
                    <i className={`signal-dot ${statusClass(stream)}`} />
                    {statusLabel(stream)}
                  </span>
                </button>
              ))
            )}
          </div>
        </section>

        <section className="section">
          <header className="section__head">
            <div className="section__head-title">
              <span className="section__num">§ 03</span>
              <span className="section__title">Selected Stream</span>
            </div>
            <span className="section__aside">Preview only</span>
          </header>

          <StreamDetails
            busy={busyId === selectedStream?.id}
            onRemove={removeStream}
            onStart={startStream}
            onStop={stopStream}
            push={push}
            stream={selectedStream}
          />
        </section>
      </aside>
    </main>
  );
}

function StreamGrid({ busyId, streams, selectedId, onSelect, onStart, onStop, onRemove }) {
  if (streams.length === 0) {
    return (
      <div className="camera__viewport rtsp__viewport rtsp__viewport--empty">
        <span className="camera__corner tr" />
        <span className="camera__corner bl" />
        <div className="camera__off">
          <Video className="camera__off-icon" />
          <span className="camera__off-title">No RTSP stream selected</span>
          <span className="camera__off-sub">Start streams below to keep them on this page</span>
        </div>
      </div>
    );
  }

  return (
    <div className="rtsp__grid">
      {streams.map((stream) => (
        <StreamStage
          busy={busyId === stream.id}
          key={stream.id}
          onRemove={onRemove}
          onSelect={onSelect}
          onStart={onStart}
          onStop={onStop}
          selected={stream.id === selectedId}
          stream={stream}
        />
      ))}
    </div>
  );
}

function StreamStage({ busy, selected, stream, onSelect, onStart, onStop, onRemove }) {
  const showFeed = stream?.cameraId && stream.running;

  return (
    <article className={`camera__viewport rtsp__tile ${selected ? 'active' : ''}`}>
      <span className="camera__corner tr" />
      <span className="camera__corner bl" />

      {showFeed ? (
        <>
          <img
            className="rtsp__feed"
            alt={`${stream.name} RTSP preview`}
            src={`${getRtspFeedUrl(stream.cameraId)}?v=${stream.feedVersion || 0}`}
          />
          <div className="camera__rec">LIVE</div>
          <div className="camera__timecode">{stream.name}</div>
        </>
      ) : (
        <>
          <div className="rtsp__preview-grid" aria-hidden="true">
            {Array.from({ length: 36 }).map((_, index) => (
              <span key={index} />
            ))}
          </div>
          <div className="rtsp__preview-copy">
            {stream.status === 'starting' ? <Loader2 /> : <Radio />}
            <strong>{stream.name}</strong>
            <span>{stream.error || stream.url}</span>
          </div>
        </>
      )}

      <div className="rtsp__tile-bar">
        <button className="rtsp__tile-name" type="button" onClick={() => onSelect(stream.id)}>
          <span className={`signal-dot ${statusClass(stream)}`} />
          <strong>{stream.name}</strong>
          <em>{stream.cameraId || statusLabel(stream)}</em>
        </button>
        <div className="rtsp__tile-actions">
          {stream.running || stream.status === 'starting' ? (
            <button disabled={busy} type="button" title="Stop stream" onClick={() => onStop(stream)}>
              {busy ? <Loader2 /> : <Square />}
            </button>
          ) : (
            <button disabled={busy} type="button" title="Start stream" onClick={() => onStart(stream)}>
              {busy ? <Loader2 /> : <Play />}
            </button>
          )}
          <button disabled={busy} type="button" title="Delete stream" onClick={() => onRemove(stream.id)}>
            <Trash2 />
          </button>
        </div>
      </div>
    </article>
  );
}

function StreamDetails({ busy, stream, onStart, onStop, onRemove, push }) {
  if (!stream) {
    return <div className="dashboard__empty dashboard__empty--compact">No stream selected</div>;
  }

  const copyUrl = async () => {
    try {
      await navigator.clipboard.writeText(stream.url);
      push?.('RTSP URL copied', 'success');
    } catch {
      push?.('Could not copy RTSP URL', 'error');
    }
  };

  return (
    <>
      <dl className="video-upload__details">
        <div className="video-upload__detail">
          <dt>Name</dt>
          <dd>{stream.name}</dd>
        </div>
        <div className="video-upload__detail">
          <dt>Camera ID</dt>
          <dd>{stream.cameraId || '--'}</dd>
        </div>
        <div className="video-upload__detail">
          <dt>RTSP URL</dt>
          <dd title={stream.url}>{stream.url}</dd>
        </div>
        <div className="video-upload__detail">
          <dt>Status</dt>
          <dd className={statusClass(stream)}>{statusLabel(stream)}</dd>
        </div>
        <div className="video-upload__detail">
          <dt>Preview frames</dt>
          <dd>{stream.frameCount || 0}</dd>
        </div>
        <div className="video-upload__detail">
          <dt>Kafka frames</dt>
          <dd>{stream.publishedCount || 0}</dd>
        </div>
        <div className="video-upload__detail">
          <dt>Added</dt>
          <dd>{formatCreatedAt(stream.createdAt)}</dd>
        </div>
        <div className="video-upload__detail">
          <dt>Model overlay</dt>
          <dd>Hidden</dd>
        </div>
      </dl>

      {stream.error && <div className="video-upload__error">{stream.error}</div>}

      <div className="rtsp__detail-actions">
        <button className="btn btn--ghost" type="button" onClick={copyUrl}>
          <Link />
          Copy URL
        </button>
        {stream.running || stream.status === 'starting' ? (
          <button className="btn btn--danger-ghost" disabled={busy} type="button" onClick={() => onStop(stream)}>
            {busy ? <Loader2 /> : <Square />}
            Stop
          </button>
        ) : (
          <button className="btn btn--primary" disabled={busy} type="button" onClick={() => onStart(stream)}>
            {busy ? <Loader2 /> : <Play />}
            Start
          </button>
        )}
        <button className="btn btn--danger-ghost" disabled={busy} type="button" onClick={() => onRemove(stream.id)}>
          <Trash2 />
          Delete
        </button>
      </div>
    </>
  );
}
