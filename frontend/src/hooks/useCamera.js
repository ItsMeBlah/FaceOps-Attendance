import { useState, useRef, useCallback, useEffect } from 'react';

/**
 * Manages webcam stream lifecycle and frame capture.
 */
export function useCamera() {
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const [active, setActive] = useState(false);
  const [error, setError] = useState(null);

  const start = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
      setActive(true);
      setError(null);
    } catch (err) {
      setError(err.message || 'Camera access denied');
      setActive(false);
    }
  }, []);

  const stop = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    if (videoRef.current) videoRef.current.srcObject = null;
    setActive(false);
  }, []);

  // Capture current video frame as JPEG blob.
  // Returns the un-mirrored raw frame, so the backend sees a normal image.
  const captureBlob = useCallback(() => {
    return new Promise((resolve) => {
      const v = videoRef.current;
      if (!v || !v.videoWidth) { resolve(null); return; }
      const c = document.createElement('canvas');
      c.width = v.videoWidth;
      c.height = v.videoHeight;
      c.getContext('2d').drawImage(v, 0, 0);
      c.toBlob((b) => resolve(b), 'image/jpeg', 0.85);
    });
  }, []);

  // Cleanup on unmount
  useEffect(() => () => stop(), [stop]);

  return { videoRef, active, error, start, stop, captureBlob };
}
