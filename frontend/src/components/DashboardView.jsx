import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { RefreshCw, UserRound } from 'lucide-react';
import { getDashboardSummary, getDashboardUserDetail } from '../services/api.js';

const EMOTIONS = [
  'anger',
  'contempt',
  'disgust',
  'fear',
  'happy',
  'neutral',
  'sad',
  'surprise',
];

const EMOTION_COLORS = {
  anger: '#ef4444',
  contempt: '#a855f7',
  disgust: '#22c55e',
  fear: '#6366f1',
  happy: '#f59e0b',
  neutral: '#94a3b8',
  sad: '#38bdf8',
  surprise: '#fb7185',
};

export default function DashboardView({ push }) {
  const [summary, setSummary] = useState(null);
  const [selectedDate, setSelectedDate] = useState('');
  const [selectedUserId, setSelectedUserId] = useState(null);
  const [detail, setDetail] = useState(null);
  const [loading, setLoading] = useState(true);
  const [detailLoading, setDetailLoading] = useState(false);
  const [error, setError] = useState('');

  const loadSummary = useCallback(async (date) => {
    setLoading(true);
    setError('');
    try {
      const data = await getDashboardSummary(14, date);
      setSummary(data);
      setSelectedDate(data.selected_date || date || '');
      setSelectedUserId((current) => {
        const people = data.people || [];
        if (people.some((person) => person.user_id === current)) return current;
        return people[0]?.user_id || null;
      });
    } catch (requestError) {
      setError(requestError.message || 'Unable to load dashboard.');
      push?.('Dashboard data unavailable', 'error', 6000);
    } finally {
      setLoading(false);
    }
  }, [push]);

  useEffect(() => {
    loadSummary();
  }, [loadSummary]);

  useEffect(() => {
    if (!selectedUserId) {
      setDetail(null);
      return;
    }

    let cancelled = false;
    setDetail(null);
    setDetailLoading(true);
    getDashboardUserDetail(selectedUserId, 20, selectedDate)
      .then((data) => {
        if (!cancelled) setDetail(data);
      })
      .catch((requestError) => {
        if (!cancelled) {
          setDetail(null);
          push?.(requestError.message || 'Unable to load person detail', 'error');
        }
      })
      .finally(() => {
        if (!cancelled) setDetailLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [push, selectedDate, selectedUserId]);

  const selectedPerson = useMemo(() => {
    const detailMatchesSelection =
      detail?.user?.user_id === selectedUserId &&
      (!detail.selected_date || detail.selected_date === selectedDate);

    return (
      (detailMatchesSelection ? detail.user : null) ||
      summary?.people?.find((person) => person.user_id === selectedUserId) ||
      null
    );
  }, [detail, selectedDate, selectedUserId, summary]);

  const handleSelectDate = useCallback((date) => {
    if (!date || date === selectedDate) return;
    loadSummary(date);
  }, [loadSummary, selectedDate]);

  const refresh = useCallback(() => {
    loadSummary(selectedDate || undefined);
  }, [loadSummary, selectedDate]);

  return (
    <main className="dashboard">
      <section className="dashboard__overview">
        <header className="section__head">
          <div className="section__head-title">
            <span className="section__num">§ 01</span>
            <span className="section__title">Daily Overview</span>
          </div>
          <button className="btn btn--text" type="button" onClick={refresh} disabled={loading}>
            <RefreshCw />
            Refresh
          </button>
        </header>

        {error && <div className="dashboard__error">{error}</div>}
        {loading && !summary ? (
          <div className="dashboard__empty">Loading dashboard data</div>
        ) : (
          <div className="dashboard__overview-grid">
            <div>
              <SummaryStrip summary={summary} />
              <DailyChart
                daily={summary?.daily || []}
                selectedDate={selectedDate}
                onSelectDate={handleSelectDate}
              />
            </div>
            <EmotionDonut selectedDay={summary?.selected_day} />
          </div>
        )}
      </section>

      <section className="dashboard__people">
        <header className="section__head">
          <div className="section__head-title">
            <span className="section__num">§ 02</span>
            <span className="section__title">People Statistics</span>
          </div>
          <span className="section__aside t-num">{formatFullDate(selectedDate)}</span>
        </header>
        <PeopleTable
          people={summary?.people || []}
          selectedUserId={selectedUserId}
          onSelect={setSelectedUserId}
        />
      </section>

      <aside className="dashboard__detail">
        <PersonDetail
          person={selectedPerson}
          detail={detail}
          loading={detailLoading}
          selectedDate={selectedDate}
        />
      </aside>
    </main>
  );
}

function SummaryStrip({ summary }) {
  const totals = summary?.totals || {};
  const selectedDay = summary?.selected_day || {};
  const selectedDate = summary?.selected_date || selectedDay.date;
  const items = [
    ['Registered', totals.registered_users ?? 0],
    ['Active this day', selectedDay.active_users ?? totals.active_selected_day ?? 0],
    ['Emotion samples', selectedDay.emotion_total ?? totals.emotion_total ?? 0],
    ['Selected date', formatDay(selectedDate)],
  ];

  return (
    <div className="dashboard__summary-strip">
      {items.map(([label, value]) => (
        <div className="dashboard__summary-item" key={label}>
          <span>{label}</span>
          <strong>{value}</strong>
        </div>
      ))}
    </div>
  );
}

function DailyChart({ daily, selectedDate, onSelectDate }) {
  const maxActive = Math.max(1, ...daily.map((item) => activeCount(item)));

  return (
    <div className="dashboard__daily-chart">
      {daily.map((item) => {
        const active = activeCount(item);
        const height = Math.max(4, (active / maxActive) * 100);
        const selected = item.date === selectedDate;
        return (
          <button
            className={`dashboard__day ${selected ? 'active' : ''}`}
            key={item.date}
            type="button"
            aria-pressed={selected}
            onClick={() => onSelectDate(item.date)}
          >
            <div className="dashboard__day-bar-wrap">
              <span
                className="dashboard__day-bar"
                style={{ height: `${height}%` }}
                title={`${active} active people`}
              />
            </div>
            <span className="dashboard__day-label">{formatDay(item.date)}</span>
            <strong className="dashboard__day-value">{active}</strong>
            <span className="dashboard__day-users">active</span>
          </button>
        );
      })}
    </div>
  );
}

function EmotionDonut({ selectedDay }) {
  const [hoveredEmotion, setHoveredEmotion] = useState(null);
  const emotions = selectedDay?.emotions || {};
  const total = EMOTIONS.reduce((sum, emotion) => sum + (emotions[emotion] || 0), 0);
  const activeValue = hoveredEmotion ? emotions[hoveredEmotion] || 0 : total;
  const activeLabel = hoveredEmotion || (total > 0 ? 'total emotions' : 'no data');
  const segments = getDonutSegments(emotions, total);

  return (
    <div className="dashboard__donut-panel">
      <div className="dashboard__subhead">Emotion on {formatFullDate(selectedDay?.date)}</div>
      <div className={`dashboard__donut ${total <= 0 ? 'dashboard__donut--empty' : ''}`}>
        <svg className="dashboard__donut-svg" viewBox="0 0 120 120" aria-label="Daily emotion chart">
          <circle className="dashboard__donut-track" cx="60" cy="60" r="46" />
          {segments.map((segment) => (
            <circle
              className={[
                'dashboard__donut-segment',
                hoveredEmotion === segment.emotion ? 'active' : '',
                hoveredEmotion && hoveredEmotion !== segment.emotion ? 'muted' : '',
              ].filter(Boolean).join(' ')}
              key={segment.emotion}
              cx="60"
              cy="60"
              r="46"
              pathLength="100"
              stroke={EMOTION_COLORS[segment.emotion]}
              strokeDasharray={`${segment.percent} ${100 - segment.percent}`}
              strokeDashoffset={-segment.start}
              tabIndex="0"
              transform="rotate(-90 60 60)"
              onBlur={() => setHoveredEmotion(null)}
              onFocus={() => setHoveredEmotion(segment.emotion)}
              onMouseEnter={() => setHoveredEmotion(segment.emotion)}
              onMouseLeave={() => setHoveredEmotion(null)}
            >
              <title>{`${segment.emotion}: ${segment.value}`}</title>
            </circle>
          ))}
        </svg>
        <div>
          <strong>{activeValue}</strong>
          <span>{activeLabel}</span>
        </div>
      </div>
      <div className="dashboard__donut-legend">
        {EMOTIONS.map((emotion) => (
          <button
            className={hoveredEmotion === emotion ? 'active' : ''}
            key={emotion}
            type="button"
            onBlur={() => setHoveredEmotion(null)}
            onFocus={() => setHoveredEmotion(emotion)}
            onMouseEnter={() => setHoveredEmotion(emotion)}
            onMouseLeave={() => setHoveredEmotion(null)}
          >
            <i style={{ background: EMOTION_COLORS[emotion] }} />
            {emotion}
            <strong>{emotions[emotion] || 0}</strong>
          </button>
        ))}
      </div>
    </div>
  );
}

function PeopleTable({ people, selectedUserId, onSelect }) {
  if (!people.length) {
    return <div className="dashboard__empty">No registered people were active on this day</div>;
  }

  return (
    <div className="dashboard__person-list">
      {people.map((person) => (
        <button
          className={`dashboard__person-row ${person.user_id === selectedUserId ? 'active' : ''}`}
          key={person.user_id}
          type="button"
          onClick={() => onSelect(person.user_id)}
        >
          <Avatar person={person} />
          <span className="dashboard__person-main">
            <strong>{person.user_name}</strong>
            <span>{person.dominant_emotion || 'no emotion data'} - last seen {formatDateTime(person.last_seen)}</span>
          </span>
          <span className="dashboard__person-metric">
            <strong>{person.emotion_total}</strong>
            <span>emotions</span>
          </span>
        </button>
      ))}
    </div>
  );
}

function PersonDetail({ person, detail, loading, selectedDate }) {
  if (!person) {
    return (
      <section className="section">
        <header className="section__head">
          <div className="section__head-title">
            <span className="section__num">§ 03</span>
            <span className="section__title">Person Detail</span>
          </div>
        </header>
        <div className="dashboard__empty">Select a person to inspect details</div>
      </section>
    );
  }

  return (
    <section className="section">
      <header className="section__head">
        <div className="section__head-title">
          <span className="section__num">§ 03</span>
          <span className="section__title">Person Detail</span>
        </div>
        <span className="section__aside">{loading ? 'Loading' : formatFullDate(selectedDate)}</span>
      </header>

      <div className="dashboard__profile">
        <Avatar person={person} large />
        <div>
          <h2>{person.user_name}</h2>
          <p>{person.user_id}</p>
        </div>
      </div>

      <div className="dashboard__detail-line">
        <span>First seen</span>
        <strong>{formatDateTime(person.first_seen)}</strong>
      </div>
      <div className="dashboard__detail-line">
        <span>Last seen</span>
        <strong>{formatDateTime(person.last_seen)}</strong>
      </div>

      <EmotionBreakdown emotions={person.emotions || {}} />
      <SessionTable sessions={detail?.recent_sessions || []} />
    </section>
  );
}

function EmotionBreakdown({ emotions }) {
  const total = Math.max(1, ...Object.values(emotions), 1);

  return (
    <div className="dashboard__emotion">
      <div className="dashboard__subhead">Emotion totals</div>
      {EMOTIONS.map((emotion) => {
        const value = emotions[emotion] || 0;
        return (
          <div className="dashboard__emotion-row" key={emotion}>
            <span>{emotion}</span>
            <div><span style={{ width: `${(value / total) * 100}%` }} /></div>
            <strong>{value}</strong>
          </div>
        );
      })}
    </div>
  );
}

function SessionTable({ sessions }) {
  return (
    <div className="dashboard__sessions">
      <div className="dashboard__subhead">Sessions on selected day</div>
      {!sessions.length ? (
        <div className="dashboard__empty dashboard__empty--compact">No sessions on this day</div>
      ) : (
        <table className="dashboard__session-table">
          <thead>
            <tr>
              <th>First seen</th>
              <th>Last seen</th>
            </tr>
          </thead>
          <tbody>
            {sessions.map((session) => (
              <tr key={session.session_id}>
                <td>{formatDateTime(session.first_seen)}</td>
                <td>{formatDateTime(session.last_seen)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

function Avatar({ person, large = false }) {
  const initials = getInitials(person?.user_name || person?.user_id || '?');
  const className = `dashboard__avatar ${large ? 'dashboard__avatar--large' : ''}`;

  if (person?.avatar_url) {
    return <img className={className} src={person.avatar_url} alt="" />;
  }

  return (
    <span className={className}>
      <UserRound />
      <strong>{initials}</strong>
    </span>
  );
}

function activeCount(item) {
  return item?.active_users ?? item?.unique_users ?? 0;
}

function getDonutSegments(emotions, total) {
  if (total <= 0) return [];

  let start = 0;
  return EMOTIONS
    .map((emotion) => {
      const value = emotions[emotion] || 0;
      const percent = (value / total) * 100;
      const segment = { emotion, value, percent, start };
      start += percent;
      return segment;
    })
    .filter((segment) => segment.value > 0);
}

function getDominantEmotion(emotions) {
  let dominant = null;
  let best = 0;
  for (const emotion of EMOTIONS) {
    const value = emotions[emotion] || 0;
    if (value > best) {
      dominant = emotion;
      best = value;
    }
  }
  return dominant;
}

function getInitials(value) {
  return value
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((part) => part[0]?.toUpperCase())
    .join('') || '?';
}

function formatDay(value) {
  if (!value) return '--';
  return new Date(`${value}T00:00:00Z`).toLocaleDateString('en-AU', {
    day: '2-digit',
    month: 'short',
  });
}

function formatFullDate(value) {
  if (!value) return '--';
  return new Date(`${value}T00:00:00Z`).toLocaleDateString('en-AU', {
    day: '2-digit',
    month: 'short',
    year: 'numeric',
  });
}

function formatDateTime(value) {
  if (!value) return '--';
  return new Date(value).toLocaleString('en-AU', {
    day: '2-digit',
    month: 'short',
    hour: '2-digit',
    minute: '2-digit',
    hour12: false,
  });
}
