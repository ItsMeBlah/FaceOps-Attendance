import { useEffect, useRef, useState } from 'react';
import { analyzeFrame, ApiError } from '../services/api';

const INTERVAL = parseInt(import.meta.env.VITE_FRAME_INTERVAL_MS, 10) || 2000;
const DEMO_PLACEHOLDERS = import.meta.env.VITE_DEMO_PLACEHOLDERS === 'true';

/**
 * When the pipeline returns no faces (backend modules unfinished),
 * emit a synthetic placeholder so the UI demonstrates the data shape
 * but is clearly marked as not-real. The flag `_placeholder` propagates
 * through the components for visible badges.
 */
function placeholderResponse(imgW = 640, imgH = 480) {
  return {
    image_width: imgW,
    image_height: imgH,
    _placeholder: true,
    faces: [{
      _placeholder: true,
      face: {
        bbox: { x: 0.30, y: 0.22, w: 0.40, h: 0.55 },
        detection_confidence: 0.00,
        crop_width: 0,
        crop_height: 0,
      },
      emotion: { label: 'neutral', confidence: 0.0, _placeholder: true },
      anti_spoofing: { label: 'real', confidence: 0.0, _placeholder: true },
      recognition: { label: 'unknown', confidence: 0.0, matched: false, _placeholder: true },
    }],
  };
}

/**
 * Polls the pipeline endpoint at a fixed interval.
 *
 * @param {() => Promise<Blob|null>} captureBlob
 * @param {boolean} enabled
 */
export function useFrameAnalysis(captureBlob, enabled) {
  const [analysis, setAnalysis] = useState(null);
  const [reqStats, setReqStats] = useState({ ok: 0, err: 0, lastErr: null });
  const [connected, setConnected] = useState(null); // null | true | false
  const consecutiveErr = useRef(0);
  const inFlight = useRef(false);
  const timerRef = useRef(null);

  useEffect(() => {
    if (!enabled) {
      if (timerRef.current) clearTimeout(timerRef.current);
      setAnalysis(null);
      return;
    }

    let cancelled = false;

    const run = async () => {
      if (cancelled || inFlight.current) {
        timerRef.current = setTimeout(run, INTERVAL);
        return;
      }
      inFlight.current = true;
      try {
        const blob = await captureBlob();
        if (!blob) {
          inFlight.current = false;
          timerRef.current = setTimeout(run, INTERVAL);
          return;
        }
        const data = await analyzeFrame(blob);

        consecutiveErr.current = 0;
        setConnected(true);

        // Inject placeholder when backend returns no faces and demo flag is on
        if (DEMO_PLACEHOLDERS && (!data.faces || data.faces.length === 0)) {
          setAnalysis({ ...placeholderResponse(data.image_width, data.image_height) });
        } else {
          setAnalysis(data);
        }
        setReqStats((s) => ({ ...s, ok: s.ok + 1, lastErr: null }));
      } catch (err) {
        consecutiveErr.current += 1;
        setConnected(false);
        setReqStats((s) => ({
          ...s,
          err: s.err + 1,
          lastErr: err.message,
        }));
        // Keep showing the previous frame's data, do NOT clear
      } finally {
        inFlight.current = false;
        if (!cancelled) timerRef.current = setTimeout(run, INTERVAL);
      }
    };

    run();
    return () => {
      cancelled = true;
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [enabled, captureBlob]);

  return { analysis, reqStats, connected };
}
