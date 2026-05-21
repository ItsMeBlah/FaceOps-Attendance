import React from 'react';
import { EmotionGlyph } from './Icons.jsx';
import { PlaceholderTag, PlaceholderBanner } from './PlaceholderBadge.jsx';

export default function DetectionPanel({ analysis }) {
  const face = analysis?.faces?.[0];
  const isPlaceholder = analysis?._placeholder === true;

  const recognition = face?.recognition;
  const emotion = face?.emotion;
  const antiSpoofing = face?.anti_spoofing;

  const noFace = !face;

  return (
    <section className="section">
      <header className="section__head">
        <div className="section__head-title">
          <span className="section__num">§ 02</span>
          <span className="section__title">The Subject</span>
        </div>
        <span className="section__aside">Live readings</span>
      </header>

      {isPlaceholder && (
        <PlaceholderBanner
          title="Placeholder readings"
          text="The detection and recognition modules are not yet wired into the pipeline. Values shown below are mock fixtures so the layout can be reviewed."
        />
      )}

      {/* IDENTITY */}
      <div className="readout">
        <div className="readout__label">
          Identity
          {recognition?._placeholder && <PlaceholderTag label="Mock" />}
        </div>
        {noFace ? (
          <div className="readout__value readout__value--mute">No subject in frame</div>
        ) : recognition?.matched && recognition.label !== 'unknown' ? (
          <>
            <div className="readout__value readout__value--accent">{recognition.label}</div>
            <ConfidenceBar value={recognition.confidence} variant="accent" />
          </>
        ) : (
          <>
            <div className="readout__value readout__value--mute">Unenrolled</div>
            <div className="readout__sub">
              <span>No match in registry</span>
              <span className="t-num">
                {recognition ? `${(recognition.confidence * 100).toFixed(1)}%` : '—'}
              </span>
            </div>
          </>
        )}
      </div>

      {/* EMOTION */}
      <div className="readout">
        <div className="readout__label">
          Affect
          {emotion?._placeholder && <PlaceholderTag label="Mock" />}
        </div>
        {noFace ? (
          <div className="readout__value readout__value--mute">—</div>
        ) : (
          <>
            <div className="readout__row">
              <EmotionGlyph emotion={emotion?.label} />
              <div style={{ flex: 1 }}>
                <div className="readout__value">
                  {capitalize(emotion?.label) || '—'}
                </div>
              </div>
            </div>
            <ConfidenceBar value={emotion?.confidence ?? 0} />
          </>
        )}
      </div>

      {/* LIVENESS */}
      <div className="readout">
        <div className="readout__label">
          Liveness
          {antiSpoofing?._placeholder && <PlaceholderTag label="Mock" />}
        </div>
        {noFace ? (
          <div className="readout__value readout__value--mute">—</div>
        ) : antiSpoofing?.label === 'fake' ? (
          <>
            <div className="readout__row">
              <span className="readout__indicator stop" />
              <span className="readout__value readout__value--stop">Spoof Detected</span>
            </div>
            <ConfidenceBar value={antiSpoofing.confidence} variant="accent" />
          </>
        ) : antiSpoofing?.label === 'real' ? (
          <>
            <div className="readout__row">
              <span className="readout__indicator go" />
              <span className="readout__value readout__value--go">Live</span>
            </div>
            <ConfidenceBar value={antiSpoofing.confidence} variant="go" />
          </>
        ) : (
          <div className="readout__value readout__value--mute">Awaiting subject</div>
        )}
      </div>
    </section>
  );
}

function capitalize(s) {
  if (!s) return '';
  return s.charAt(0).toUpperCase() + s.slice(1).toLowerCase();
}

function ConfidenceBar({ value = 0, variant }) {
  const pct = Math.max(0, Math.min(1, value));
  const fillClass = variant ? `readout__sub-bar-fill--${variant}` : '';
  return (
    <div className="readout__sub">
      <span>Confidence</span>
      <span className="readout__sub-bar">
        <span
          className={`readout__sub-bar-fill ${fillClass}`}
          style={{ width: `${pct * 100}%` }}
        />
      </span>
      <span className="t-num">{(pct * 100).toFixed(1)}%</span>
    </div>
  );
}
