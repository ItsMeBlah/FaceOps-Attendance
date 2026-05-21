import React from 'react';

export default function AttendanceLog({ entries }) {
  return (
    <section className="section">
      <header className="section__head">
        <div className="section__head-title">
          <span className="section__num">§ 04</span>
          <span className="section__title">Attendance Log</span>
        </div>
        <span className="section__aside t-num">{String(entries.length).padStart(3, '0')} entries</span>
      </header>

      <div className="log">
        {entries.map((e) => (
          <div className="log__row" key={e.id}>
            <span className="log__time">{e.time}</span>
            <span className="log__name">{e.name}</span>
            {!e.isLive ? (
              <span className="log__badge log__badge--spoof">Spoof</span>
            ) : e.isNew ? (
              <span className="log__badge log__badge--new">New</span>
            ) : (
              <span style={{ width: 0 }} />
            )}
          </div>
        ))}
      </div>
    </section>
  );
}
