/**
 * Draw bounding boxes + labels on the overlay canvas.
 * Designed to match the editorial aesthetic — thin precise marks,
 * mono-typeface labels, accent color for alerts.
 */

const PAPER = '#EFE9DD';
const ACCENT = '#B53A1F';
const INK = '#1F1A16';
const SIGNAL_GO = '#4A6B3A';

export function clearCanvas(ctx) {
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
}

/**
 * @param {CanvasRenderingContext2D} ctx
 * @param {Array} faces  - from FrameAnalysisResponse.faces
 * @param {number} imgW
 * @param {number} imgH
 * @param {boolean} mirrored - whether display is CSS-mirrored
 */
export function drawFaces(ctx, faces, imgW, imgH, mirrored = true) {
  const canvas = ctx.canvas;
  clearCanvas(ctx);
  if (!faces?.length || !imgW || !imgH) return;

  // Compute scale to match object-fit: cover of the video element
  const vidAspect = imgW / imgH;
  const canAspect = canvas.width / canvas.height;
  let scale, offsetX = 0, offsetY = 0;

  if (canAspect > vidAspect) {
    scale = canvas.width / imgW;
    offsetY = (canvas.height - imgH * scale) / 2;
  } else {
    scale = canvas.height / imgH;
    offsetX = (canvas.width - imgW * scale) / 2;
  }

  for (const face of faces) {
    const bbox = face.face?.bbox;
    if (!bbox) continue;

    const isNormalized =
      bbox.x <= 1.0 && bbox.y <= 1.0 && bbox.w <= 1.0 && bbox.h <= 1.0;

    let bx, by, bw, bh;
    if (isNormalized) {
      bx = bbox.x * imgW * scale + offsetX;
      by = bbox.y * imgH * scale + offsetY;
      bw = bbox.w * imgW * scale;
      bh = bbox.h * imgH * scale;
    } else {
      bx = bbox.x * scale + offsetX;
      by = bbox.y * scale + offsetY;
      bw = bbox.w * scale;
      bh = bbox.h * scale;
    }

    if (mirrored) bx = canvas.width - bx - bw;

    const isSpoof = face.anti_spoofing?.label === 'fake';
    const isPlaceholder = face._placeholder === true;
    const color = isSpoof ? ACCENT : (isPlaceholder ? '#B07A0A' : PAPER);

    // Corner brackets — thinner, more refined
    const corner = Math.min(20, bw * 0.18, bh * 0.18);
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.lineCap = 'butt';
    ctx.setLineDash(isPlaceholder ? [4, 3] : []);

    const c = (x, y, dx1, dy1, dx2, dy2) => {
      ctx.beginPath();
      ctx.moveTo(x + dx1, y + dy1);
      ctx.lineTo(x, y);
      ctx.lineTo(x + dx2, y + dy2);
      ctx.stroke();
    };
    c(bx, by, 0, corner, corner, 0);
    c(bx + bw, by, -corner, 0, 0, corner);
    c(bx, by + bh, 0, -corner, corner, 0);
    c(bx + bw, by + bh, -corner, 0, 0, -corner);

    ctx.setLineDash([]);

    // Identity label — small mono caps above the box
    const name = face.recognition?.label;
    const matched = face.recognition?.matched;
    if (name) {
      const text = (matched && name !== 'unknown') ? name.toUpperCase() : 'UNIDENTIFIED';
      ctx.font = '500 10px "IBM Plex Mono", monospace';
      ctx.fillStyle = color;
      const padding = 6;
      const tw = ctx.measureText(text).width;
      // Background bar
      ctx.fillRect(bx, by - 22, tw + padding * 2, 18);
      ctx.fillStyle = (color === PAPER) ? INK : PAPER;
      ctx.fillText(text, bx + padding, by - 9);
    }

    // Confidence tick at bottom-right of box (mono numerals)
    const conf = face.face?.detection_confidence;
    if (typeof conf === 'number' && conf > 0) {
      ctx.font = '500 9px "IBM Plex Mono", monospace';
      ctx.fillStyle = color;
      const txt = `${(conf * 100).toFixed(0)}%`;
      const tw = ctx.measureText(txt).width;
      ctx.fillText(txt, bx + bw - tw - 2, by + bh - 6);
    }

    // Placeholder ribbon for mock data
    if (isPlaceholder) {
      ctx.font = '600 9px "IBM Plex Mono", monospace';
      ctx.fillStyle = '#B07A0A';
      ctx.fillText('// PLACEHOLDER DATA', bx, by + bh + 14);
    }
  }
}
