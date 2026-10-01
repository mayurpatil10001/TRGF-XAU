// src/components/SignalPanel.tsx
// Last signal with 3-bar probability display and reason text

import type { LastSignal } from '../types';

interface Props { signal: LastSignal | null; threshold: number; }

function ProbBar({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 10 }}>
      <div style={{ width: 70, fontSize: 11, fontWeight: 600, color: 'var(--text-secondary)', textAlign: 'right' }}>
        {label}
      </div>
      <div style={{ flex: 1, height: 20, background: 'rgba(255,255,255,0.05)', borderRadius: 4, overflow: 'hidden', position: 'relative' }}>
        <div style={{
          width: `${(value * 100).toFixed(1)}%`, height: '100%',
          background: color, borderRadius: 4,
          transition: 'width 0.4s ease',
        }} />
        {/* Threshold marker */}
        <div style={{
          position: 'absolute', top: 0, bottom: 0, left: `${(threshold * 100)}%`,
          width: 2, background: 'rgba(255,255,255,0.4)',
        }} />
      </div>
      <div className="tabular" style={{ width: 48, fontSize: 12, fontWeight: 600, color, textAlign: 'right' }}>
        {(value * 100).toFixed(1)}%
      </div>
    </div>
  );
}

const threshold = 0.55; // passed via prop ideally

export function SignalPanel({ signal, threshold }: Props) {
  if (!signal) return (
    <div className="card">
      <div style={{ color: 'var(--text-muted)', fontSize: 13 }}>No signal yet.</div>
    </div>
  );

  const actionColor: Record<string, string> = {
    BUY: 'var(--green)', SELL: 'var(--red)', NO_TRADE: 'var(--text-muted)',
  };

  return (
    <div className="card">
      <div style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 12, fontWeight: 600, letterSpacing: '0.8px', textTransform: 'uppercase' }}>
        📡 Last Signal
      </div>

      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 16 }}>
        <div>
          <div style={{
            fontSize: 28, fontWeight: 900,
            color: actionColor[signal.action] ?? 'var(--text-primary)',
            letterSpacing: '-0.5px',
          }}>
            {signal.action}
          </div>
          <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 4 }}>
            {signal.bar_time ? new Date(signal.bar_time).toLocaleTimeString('en-GB') : '—'}
          </div>
        </div>

        {signal.action === 'NO_TRADE' && signal.reason && (
          <div style={{
            background: 'rgba(255,255,255,0.04)', borderRadius: 8,
            padding: '6px 10px', fontSize: 11, color: 'var(--text-muted)',
            maxWidth: 180, wordBreak: 'break-word',
          }}>
            {signal.reason.replace(/_/g, ' ')}
          </div>
        )}
      </div>

      <ProbBar label="NO TRADE" value={signal.p_none} color="var(--text-muted)" />
      <ProbBar label="LONG" value={signal.p_long} color="var(--green)" />
      <ProbBar label="SHORT" value={signal.p_short} color="var(--red)" />

      <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 8 }}>
        Threshold: <span style={{ color: 'var(--accent)' }} className="tabular">{(threshold * 100).toFixed(0)}%</span>
      </div>
    </div>
  );
}
