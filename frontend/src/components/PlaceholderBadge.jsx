import React from 'react';
import { AlertIcon } from './Icons.jsx';

/** Inline chip indicating mock / non-real data */
export const PlaceholderTag = ({ label = 'Placeholder' }) => (
  <span className="placeholder-tag" title="Mock data — backend module not yet integrated">
    {label}
  </span>
);

/** Full-width banner for top of a section */
export const PlaceholderBanner = ({ title = 'Demonstration mode', text }) => (
  <div className="placeholder-banner" role="status">
    <AlertIcon className="placeholder-banner__icon" width="16" height="16" />
    <div>
      <div className="placeholder-banner__title">{title}</div>
      <div className="placeholder-banner__text">{text}</div>
    </div>
  </div>
);
