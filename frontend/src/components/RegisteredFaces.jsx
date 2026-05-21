import React from 'react';

export default function RegisteredFaces({ entries, selectedName, onSelect, onRemove }) {
  const names = Object.keys(entries).sort();

  return (
    <section className="section">
      <header className="section__head">
        <div className="section__head-title">
          <span className="section__num">§ 03</span>
          <span className="section__title">Registered Faces</span>
        </div>
        <span className="section__aside t-num">{String(names.length).padStart(2, '0')} on file</span>
      </header>

      <ul className="reg-list">
        {names.map((name, i) => (
          <li
            key={name}
            className={`reg-item ${selectedName === name ? 'selected' : ''}`}
            onClick={() => onSelect(name)}
          >
            <span className="reg-item__index">№ {String(i + 1).padStart(2, '0')}</span>
            <span className="reg-item__name">{name}</span>
            <span className="reg-item__time">
              {entries[name].lastSeen ? `seen ${entries[name].lastSeen}` : 'never seen'}
            </span>
          </li>
        ))}
      </ul>

      <div className="reg-actions">
        <button
          className="btn btn--text"
          disabled={!selectedName}
          onClick={onRemove}
        >
          Remove face
        </button>
      </div>
    </section>
  );
}
