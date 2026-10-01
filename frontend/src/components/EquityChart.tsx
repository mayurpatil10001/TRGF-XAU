// src/components/EquityChart.tsx
// Streaming equity/balance/realized PnL chart using Recharts

import { useState, useEffect } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine,
} from 'recharts';
import type { LiveSnapshot } from '../types';

interface ChartPoint {
  ts: string;
  equity: number;
  balance: number;
  realized: number;
}

type Series = 'equity' | 'balance' | 'realized';
type Range = '15m' | '1h' | '1d' | 'all';

interface Props { snapshot: LiveSnapshot | null; }

const MAX_POINTS = 7200; // 2 hours at 1s

function filterByRange(points: ChartPoint[], range: Range): ChartPoint[] {
  if (range === 'all') return points;
  const now = Date.now();
  const cutoff: Record<Range, number> = {
    '15m': 15 * 60 * 1000,
    '1h': 60 * 60 * 1000,
    '1d': 24 * 60 * 60 * 1000,
    'all': 0,
  };
  return points.filter(p => now - new Date(p.ts).getTime() < cutoff[range]);
}

export function EquityChart({ snapshot }: Props) {
  const [points, setPoints] = useState<ChartPoint[]>([]);
  const [activeSeries, setActiveSeries] = useState<Series[]>(['equity']);
  const [range, setRange] = useState<Range>('1h');

  // Append new point from snapshot
  useEffect(() => {
    if (!snapshot || snapshot.equity === 0) return;
    setPoints(prev => {
      const newPt: ChartPoint = {
        ts: snapshot.ts,
        equity: snapshot.equity,
        balance: snapshot.balance,
        realized: snapshot.realized_pnl_total,
      };
      const next = [...prev, newPt];
      return next.length > MAX_POINTS ? next.slice(-MAX_POINTS) : next;
    });
  }, [snapshot?.ts]);

  // Load history on mount
  useEffect(() => {
    fetch('/api/equity_curve?limit=5000')
      .then(r => r.json())
      .then((data: Array<{ ts: string; equity: number; balance: number; realized_pnl: number }>) => {
        setPoints(data.map(d => ({ ts: d.ts, equity: d.equity, balance: d.balance, realized: d.realized_pnl })));
      })
      .catch(() => {});
  }, []);

  const visible = filterByRange(points, range);

  const SERIES_CONFIG = {
    equity: { color: 'var(--blue)', label: 'Equity' },
    balance: { color: 'var(--green)', label: 'Balance' },
    realized: { color: 'var(--purple)', label: 'Realized P/L' },
  };

  const toggleSeries = (s: Series) => {
    setActiveSeries(prev =>
      prev.includes(s) ? (prev.length > 1 ? prev.filter(x => x !== s) : prev) : [...prev, s]
    );
  };

  const formatTs = (ts: string) => {
    try { return new Date(ts).toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' }); }
    catch { return ts; }
  };

  const baselineBalance = points[0]?.balance ?? 0;

  return (
    <div className="card" style={{ padding: '20px 16px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
        <span style={{ fontWeight: 600, fontSize: 14 }}>📈 Equity Curve</span>

        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          {/* Series toggles */}
          {(Object.entries(SERIES_CONFIG) as [Series, { color: string; label: string }][]).map(([s, conf]) => (
            <button key={s}
              onClick={() => toggleSeries(s)}
              style={{
                background: activeSeries.includes(s) ? conf.color + '22' : 'transparent',
                color: activeSeries.includes(s) ? conf.color : 'var(--text-muted)',
                border: `1px solid ${activeSeries.includes(s) ? conf.color + '66' : 'var(--border)'}`,
                borderRadius: 6, padding: '3px 10px', cursor: 'pointer', fontSize: 11, fontWeight: 600,
              }}>
              {conf.label}
            </button>
          ))}

          {/* Range selector */}
          {(['15m', '1h', '1d', 'all'] as Range[]).map(r => (
            <button key={r}
              onClick={() => setRange(r)}
              style={{
                background: range === r ? 'var(--accent-glow)' : 'transparent',
                color: range === r ? 'var(--accent)' : 'var(--text-muted)',
                border: `1px solid ${range === r ? 'var(--border-accent)' : 'var(--border)'}`,
                borderRadius: 6, padding: '3px 8px', cursor: 'pointer', fontSize: 11, fontWeight: 600,
              }}>
              {r}
            </button>
          ))}
        </div>
      </div>

      <ResponsiveContainer width="100%" height={220}>
        <LineChart data={visible} margin={{ top: 4, right: 4, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
          <XAxis dataKey="ts" tickFormatter={formatTs} tick={{ fill: '#555d75', fontSize: 10 }} minTickGap={80} />
          <YAxis tick={{ fill: '#555d75', fontSize: 10 }} width={72} tickFormatter={v => v.toLocaleString('en-US', { maximumFractionDigits: 0 })} />
          <Tooltip
            contentStyle={{ background: '#13151d', border: '1px solid rgba(255,255,255,0.08)', borderRadius: 8 }}
            labelFormatter={(label: unknown) => formatTs(String(label))}
            formatter={(v: unknown) => (typeof v === 'number' ? v.toFixed(2) : String(v))}
          />
          {baselineBalance > 0 && (
            <ReferenceLine y={baselineBalance} stroke="rgba(255,255,255,0.1)" strokeDasharray="4 4" label={{ value: 'Start', fill: '#555', fontSize: 10 }} />
          )}
          {activeSeries.includes('equity') && (
            <Line type="monotone" dataKey="equity" stroke="#60a5fa" strokeWidth={2} dot={false} name="Equity" />
          )}
          {activeSeries.includes('balance') && (
            <Line type="monotone" dataKey="balance" stroke="#34d399" strokeWidth={1.5} dot={false} name="Balance" strokeDasharray="4 4" />
          )}
          {activeSeries.includes('realized') && (
            <Line type="monotone" dataKey="realized" stroke="#a78bfa" strokeWidth={1.5} dot={false} name="Realized P/L" />
          )}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
