import React from 'react';

const baseProps = {
  viewBox: '0 0 48 48',
  fill: 'none',
  stroke: 'currentColor',
  strokeWidth: '1.25',
  strokeLinecap: 'round',
  strokeLinejoin: 'round',
};

export const EmotionGlyph = ({ emotion = 'neutral' }) => {
  const e = (emotion || '').toLowerCase();
  const glyphs = {
    neutral: (
      <svg {...baseProps}><circle cx="24" cy="24" r="20"/><line x1="14" y1="20" x2="18" y2="20"/><line x1="30" y1="20" x2="34" y2="20"/><line x1="16" y1="30" x2="32" y2="30"/></svg>
    ),
    happy: (
      <svg {...baseProps}><circle cx="24" cy="24" r="20"/><line x1="14" y1="20" x2="18" y2="20"/><line x1="30" y1="20" x2="34" y2="20"/><path d="M14 28 Q24 36 34 28"/></svg>
    ),
    sad: (
      <svg {...baseProps}><circle cx="24" cy="24" r="20"/><line x1="14" y1="20" x2="18" y2="20"/><line x1="30" y1="20" x2="34" y2="20"/><path d="M14 32 Q24 24 34 32"/></svg>
    ),
    angry: (
      <svg {...baseProps}><circle cx="24" cy="24" r="20"/><line x1="13" y1="18" x2="19" y2="22"/><line x1="35" y1="18" x2="29" y2="22"/><path d="M14 32 Q24 26 34 32"/></svg>
    ),
    surprise: (
      <svg {...baseProps}><circle cx="24" cy="24" r="20"/><circle cx="16" cy="20" r="2"/><circle cx="32" cy="20" r="2"/><circle cx="24" cy="30" r="3.5"/></svg>
    ),
    surprised: (
      <svg {...baseProps}><circle cx="24" cy="24" r="20"/><circle cx="16" cy="20" r="2"/><circle cx="32" cy="20" r="2"/><circle cx="24" cy="30" r="3.5"/></svg>
    ),
    fear: (
      <svg {...baseProps}><circle cx="24" cy="24" r="20"/><line x1="14" y1="18" x2="18" y2="22"/><line x1="34" y1="18" x2="30" y2="22"/><path d="M18 31 Q20 28 22 31 Q24 28 26 31 Q28 28 30 31"/></svg>
    ),
    disgust: (
      <svg {...baseProps}><circle cx="24" cy="24" r="20"/><line x1="14" y1="22" x2="18" y2="20"/><line x1="30" y1="20" x2="34" y2="22"/><path d="M16 30 L22 30 L24 32 L26 30 L32 30"/></svg>
    ),
    contempt: (
      <svg {...baseProps}><circle cx="24" cy="24" r="20"/><line x1="14" y1="20" x2="18" y2="20"/><line x1="30" y1="20" x2="34" y2="20"/><path d="M16 30 L28 30 L32 28"/></svg>
    ),
  };
  return <span className="emotion-glyph" aria-label={emotion}>{glyphs[e] || glyphs.neutral}</span>;
};

/* ── Utility icons (small, used in buttons / chips) ────────── */

export const PlayIcon = (p) => <svg viewBox="0 0 16 16" fill="currentColor" {...p}><path d="M4 3 L13 8 L4 13 Z"/></svg>;
export const PauseIcon = (p) => <svg viewBox="0 0 16 16" fill="currentColor" {...p}><rect x="4" y="3" width="3" height="10"/><rect x="9" y="3" width="3" height="10"/></svg>;
export const UploadIcon = (p) => <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" {...p}><path d="M8 11 V3 M5 6 L8 3 L11 6"/><path d="M3 10 V13 H13 V10"/></svg>;
export const DownloadIcon = (p) => <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" {...p}><path d="M8 3 V11 M5 8 L8 11 L11 8"/><path d="M3 10 V13 H13 V10"/></svg>;
export const CameraOffIcon = (p) => (
  <svg viewBox="0 0 56 56" fill="none" stroke="currentColor" strokeWidth="1.25" {...p}>
    <rect x="6" y="14" width="44" height="32" />
    <circle cx="28" cy="30" r="9" />
    <line x1="8" y1="8" x2="48" y2="50" />
  </svg>
);
