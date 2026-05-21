/**
 * Hardcoded center crop helpers, per teammate's note:
 *
 * "Currently we don't have face detection so just hardcode the x,y,w,h
 *  in the frontend first to draw the box, and crop the frame inside
 *  that box to send to emotion and spoof routes."
 *
 * The box is defined in normalized (0–1) coordinates so it scales
 * with any camera resolution.
 */

// Centered box, ~45% width, ~62% height — covers a typical face position
export const HARDCODED_BOX = {
  x: 0.275, // left edge
  y: 0.14,  // top edge
  w: 0.45,
  h: 0.62,
};

/**
 * Crop the source canvas/video into a new blob using normalized bbox.
 * @returns {Promise<{blob: Blob, dataUrl: string, pixelBox: {x,y,w,h}}>}
 */
export async function cropToBlob(sourceEl, box = HARDCODED_BOX) {
  const w = sourceEl.videoWidth || sourceEl.width;
  const h = sourceEl.videoHeight || sourceEl.height;
  if (!w || !h) throw new Error('Source has no dimensions yet');

  const px = Math.floor(box.x * w);
  const py = Math.floor(box.y * h);
  const pw = Math.floor(box.w * w);
  const ph = Math.floor(box.h * h);

  const c = document.createElement('canvas');
  c.width = pw;
  c.height = ph;
  const ctx = c.getContext('2d');
  ctx.drawImage(sourceEl, px, py, pw, ph, 0, 0, pw, ph);

  const dataUrl = c.toDataURL('image/jpeg', 0.85);
  const blob = await new Promise((res) => c.toBlob(res, 'image/jpeg', 0.85));
  return { blob, dataUrl, pixelBox: { x: px, y: py, w: pw, h: ph } };
}

/** Full frame as blob (no cropping) */
export async function frameToBlob(sourceEl) {
  const w = sourceEl.videoWidth || sourceEl.width;
  const h = sourceEl.videoHeight || sourceEl.height;
  if (!w || !h) throw new Error('Source has no dimensions yet');
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  c.getContext('2d').drawImage(sourceEl, 0, 0);
  return new Promise((res) => c.toBlob(res, 'image/jpeg', 0.85));
}
