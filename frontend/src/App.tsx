// src/App.tsx
// Main dashboard app

import { useState, useCallback } from 'react';
import './index.css';
import { useLiveSocket } from './hooks/useLiveSocket';
import { KpiCard } from './components/KpiCard';
import { Header } from './components/Header';
import { EquityChart } from './components/EquityChart';
import { SignalPanel } from './components/SignalPanel';
import { PositionsTable } from './components/PositionsTable';
import { KillSwitchModal } from './components/KillSwitchModal';

const WS_URL = `ws://${window.location.host}/ws/live`;

export default function App() {
  const { snapshot, wsState, stale } = useLiveSocket(WS_URL);
  const [showKillModal, setShowKillModal] = useState(false);
  const [presentationMode, setPresentationMode] = useState(false);

  const handleKillConfirm = useCallback(async () => {
    const token = prompt('Enter API token:') ?? '';
    await fetch('/api/kill_switch', {
      method: 'POST',
      headers: { 'X-API-Token': token },
    });
    setShowKillModal(false);
  }, []);

  const s = snapshot;
  const todayReturn = s && s.balance > 0
    ? ((s.equity - s.balance) / s.balance) * 100
    : 0;

  return (
    <div style={{ minHeight: '100vh', background: 'var(--bg-primary)', fontSize: presentationMode ? 18 : 14 }}>
      {/* Stale banner — ONLY when stale; numbers grayed out */}
      {stale && (
        <div className="stale-banner">
          ⚠ STALE — NOT LIVE · WebSocket {wsState}
        </div>
      )}

      <Header
        snapshot={s}
        wsState={wsState}
        onKillSwitch={() => setShowKillModal(true)}
      />

      <main style={{
        maxWidth: 1600, margin: '0 auto', padding: '24px 20px',
        paddingTop: stale ? 56 : 24,
        filter: stale ? 'grayscale(60%) opacity(0.6)' : 'none',
        transition: 'filter 0.3s',
      }}>

        {/* Presentation Mode toggle */}
        <div style={{ display: 'flex', justifyContent: 'flex-end', marginBottom: 16 }}>
          <button
            onClick={() => setPresentationMode(p => !p)}
            style={{
              background: presentationMode ? 'var(--accent-glow)' : 'transparent',
              color: presentationMode ? 'var(--accent)' : 'var(--text-muted)',
              border: `1px solid ${presentationMode ? 'var(--border-accent)' : 'var(--border)'}`,
              borderRadius: 8, padding: '5px 14px', cursor: 'pointer', fontSize: 12,
            }}>
            {presentationMode ? '🖥 Presentation ON' : '🖥 Presentation Mode'}
          </button>
        </div>

        {/* KPI Row */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))',
          gap: 12, marginBottom: 20,
        }}>
          <KpiCard id="kpi-balance" label="Balance" value={s?.balance ?? 0} prefix="$" color="neutral" />
          <KpiCard id="kpi-equity" label="Equity" value={s?.equity ?? 0} prefix="$" color={s && s.equity >= s.balance ? 'green' : 'red'} />
          <KpiCard id="kpi-floating" label="Floating P/L" value={s?.floating_pnl ?? 0} prefix="$" />
          <KpiCard id="kpi-today" label="Today Realized" value={s?.realized_pnl_today ?? 0} prefix="$" />
          <KpiCard id="kpi-total" label="Total Realized" value={s?.realized_pnl_total ?? 0} prefix="$" />
          <KpiCard id="kpi-return" label="Today Return" value={todayReturn} suffix="%" small />
          <KpiCard id="kpi-freemargin" label="Free Margin" value={s?.free_margin ?? 0} prefix="$" color="neutral" />
        </div>

        {/* Market row */}
        <div style={{ display: 'flex', gap: 12, marginBottom: 20, flexWrap: 'wrap' }}>
          <div className="card" style={{ display: 'flex', gap: 24, padding: '14px 20px', flex: '0 0 auto' }}>
            <div>
              <div style={{ fontSize: 10, color: 'var(--text-muted)', fontWeight: 600, letterSpacing: '0.8px', marginBottom: 4 }}>BID</div>
              <div className="tabular" style={{ fontSize: 22, fontWeight: 700 }}>{(s?.market?.bid ?? 0).toFixed(2)}</div>
            </div>
            <div>
              <div style={{ fontSize: 10, color: 'var(--text-muted)', fontWeight: 600, letterSpacing: '0.8px', marginBottom: 4 }}>ASK</div>
              <div className="tabular" style={{ fontSize: 22, fontWeight: 700 }}>{(s?.market?.ask ?? 0).toFixed(2)}</div>
            </div>
            <div>
              <div style={{ fontSize: 10, color: 'var(--text-muted)', fontWeight: 600, letterSpacing: '0.8px', marginBottom: 4 }}>SPREAD</div>
              <div className="tabular" style={{ fontSize: 22, fontWeight: 700, color: 'var(--yellow)' }}>{(s?.market?.spread ?? 0).toFixed(3)}</div>
            </div>
          </div>

          {/* Bot stats */}
          <div className="card" style={{ display: 'flex', gap: 20, padding: '14px 20px', flex: '0 0 auto' }}>
            <div>
              <div style={{ fontSize: 10, color: 'var(--text-muted)', fontWeight: 600, letterSpacing: '0.8px', marginBottom: 4 }}>DAILY LOSS</div>
              <div className="tabular" style={{ fontSize: 18, fontWeight: 700, color: (s?.bot?.daily_loss_used_pct ?? 0) > 2 ? 'var(--red)' : 'var(--text-primary)' }}>
                {(s?.bot?.daily_loss_used_pct ?? 0).toFixed(1)}%
              </div>
            </div>
            <div>
              <div style={{ fontSize: 10, color: 'var(--text-muted)', fontWeight: 600, letterSpacing: '0.8px', marginBottom: 4 }}>CONSEC. LOSSES</div>
              <div className="tabular" style={{ fontSize: 18, fontWeight: 700, color: (s?.bot?.consecutive_losses ?? 0) >= 2 ? 'var(--red)' : 'var(--text-primary)' }}>
                {s?.bot?.consecutive_losses ?? 0}
              </div>
            </div>
          </div>
        </div>

        {/* Chart + Signal */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 320px', gap: 16, marginBottom: 20 }}>
          <EquityChart snapshot={s} />
          <SignalPanel signal={s?.last_signal ?? null} threshold={s?.last_signal?.threshold ?? 0.55} />
        </div>

        {/* Positions */}
        <div style={{ marginBottom: 20 }}>
          <PositionsTable positions={s?.open_positions ?? []} horizon={3} />
        </div>
      </main>

      {showKillModal && (
        <KillSwitchModal
          onConfirm={handleKillConfirm}
          onCancel={() => setShowKillModal(false)}
        />
      )}
    </div>
  );
}
