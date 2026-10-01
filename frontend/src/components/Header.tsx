// src/components/Header.tsx
import type { LiveSnapshot } from '../types';

interface HeaderProps {
  snapshot: LiveSnapshot | null;
  wsState: string;
  onKillSwitch: () => void;
}

const WS_COLORS: Record<string, string> = {
  LIVE: 'var(--green)',
  CONNECTING: 'var(--yellow)',
  STALE: 'var(--red)',
  RECONNECTING: 'var(--orange)',
};

const GATE_PILLS: Record<string, { cls: string; label: string }> = {
  PASSED:      { cls: 'pill-green',  label: '✓ Gate PASSED' },
  PROVISIONAL: { cls: 'pill-yellow', label: '~ Gate PROVISIONAL' },
  FAILED:      { cls: 'pill-red',    label: '✗ Gate FAILED' },
  UNKNOWN:     { cls: 'pill-blue',   label: '? Gate UNKNOWN' },
};

export function Header({ snapshot, wsState, onKillSwitch }: HeaderProps) {
  const acc = snapshot?.account;
  const gateKey = snapshot?.profit_gate?.status ?? 'UNKNOWN';
  const gatePill = GATE_PILLS[gateKey] ?? GATE_PILLS['UNKNOWN'];
  const botStatus = snapshot?.bot?.status ?? 'UNKNOWN';

  const botColor: Record<string, string> = {
    RUNNING: 'var(--green)', PAUSED: 'var(--yellow)',
    SIGNAL_ONLY: 'var(--orange)', KILLED: 'var(--red)',
  };

  return (
    <header style={{
      display: 'flex', alignItems: 'center', justifyContent: 'space-between',
      padding: '14px 24px',
      background: 'var(--bg-secondary)',
      borderBottom: '1px solid var(--border)',
      position: 'sticky', top: 0, zIndex: 100,
    }}>
      {/* Left: Logo + status */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
        {/* MT5 dot */}
        <div className={`pulse-dot ${snapshot?.mt5_connected ? 'green' : 'red'}`} title={snapshot?.mt5_connected ? 'MT5 Connected' : 'MT5 Disconnected'} />

        <div style={{ fontSize: 18, fontWeight: 900, letterSpacing: '-0.5px' }}>
          <span style={{ background: 'var(--gradient-gold)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
            TEGF
          </span>
          <span style={{ color: 'var(--text-secondary)', fontWeight: 400 }}> · XAU/USD</span>
        </div>

        {/* WS state */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 11, color: WS_COLORS[wsState] ?? 'gray', fontWeight: 600 }}>
          <div style={{ width: 6, height: 6, borderRadius: '50%', background: WS_COLORS[wsState] ?? 'gray' }} />
          {wsState}
        </div>

        {/* Account info */}
        {acc && (
          <div style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
            #{acc.login} · {acc.server}
          </div>
        )}

        {/* DEMO/LIVE badge */}
        {acc && (
          <span className={acc.mode === 'LIVE' ? 'pill pill-red' : 'pill pill-blue'}
            style={acc.mode === 'LIVE' ? { border: '2px solid var(--red)' } : {}}>
            {acc.mode}
          </span>
        )}
      </div>

      {/* Right: Gate + Bot status + Kill */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
        <span className={`pill ${gatePill.cls}`}>{gatePill.label}</span>

        <span className="pill" style={{ background: 'rgba(0,0,0,0.3)', color: botColor[botStatus] ?? 'gray', border: `1px solid ${botColor[botStatus] ?? 'gray'}44` }}>
          {botStatus}
        </span>

        <button
          id="btn-kill-switch"
          onClick={onKillSwitch}
          style={{
            background: 'var(--red-dim)', color: 'var(--red)',
            border: '1px solid rgba(248,113,113,0.3)',
            borderRadius: 8, padding: '6px 14px', cursor: 'pointer',
            fontSize: 12, fontWeight: 700, letterSpacing: '0.5px',
            transition: 'all 0.2s',
          }}
          onMouseEnter={e => (e.currentTarget.style.background = 'rgba(248,113,113,0.25)')}
          onMouseLeave={e => (e.currentTarget.style.background = 'var(--red-dim)')}
        >
          ⚠ KILL
        </button>
      </div>
    </header>
  );
}
