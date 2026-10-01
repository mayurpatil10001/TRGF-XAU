// src/components/PositionsTable.tsx
// Live positions table with floating P/L, bars-held countdown

import type { OpenPosition } from '../types';

interface Props {
  positions: OpenPosition[];
  horizon: number;
}

export function PositionsTable({ positions, horizon }: Props) {
  if (positions.length === 0) {
    return (
      <div className="card" style={{ color: 'var(--text-muted)', fontSize: 13 }}>
        No open positions.
      </div>
    );
  }

  return (
    <div className="card" style={{ padding: '16px 0', overflow: 'hidden' }}>
      <div style={{ padding: '0 20px 12px', fontWeight: 600, fontSize: 13, letterSpacing: '0.3px' }}>
        📊 Open Positions
      </div>
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead>
            <tr style={{ background: 'rgba(255,255,255,0.03)' }}>
              {['Ticket', 'Side', 'Lots', 'Entry', 'Current', 'SL', 'TP', 'P/L', 'Bars'].map(h => (
                <th key={h} style={{ padding: '8px 14px', textAlign: h === 'Side' ? 'center' : 'right', color: 'var(--text-muted)', fontWeight: 600, letterSpacing: '0.5px', fontSize: 10, textTransform: 'uppercase', whiteSpace: 'nowrap', borderBottom: '1px solid var(--border)' }}>
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {positions.map(pos => (
              <tr key={pos.ticket} style={{ borderBottom: '1px solid rgba(255,255,255,0.04)', transition: 'background 0.15s' }}
                onMouseEnter={e => (e.currentTarget.style.background = 'rgba(255,255,255,0.02)')}
                onMouseLeave={e => (e.currentTarget.style.background = 'transparent')}>
                <td className="tabular" style={{ padding: '10px 14px', color: 'var(--text-muted)' }}>{pos.ticket}</td>
                <td style={{ padding: '10px 14px', textAlign: 'center' }}>
                  <span className={`pill ${pos.side === 'BUY' ? 'pill-green' : 'pill-red'}`}>{pos.side}</span>
                </td>
                <td className="tabular" style={{ padding: '10px 14px', textAlign: 'right' }}>{pos.lots.toFixed(2)}</td>
                <td className="tabular" style={{ padding: '10px 14px', textAlign: 'right' }}>{pos.entry.toFixed(2)}</td>
                <td className="tabular" style={{ padding: '10px 14px', textAlign: 'right' }}>{pos.price.toFixed(2)}</td>
                <td className="tabular" style={{ padding: '10px 14px', textAlign: 'right', color: 'var(--red)', fontSize: 11 }}>{pos.sl.toFixed(2)}</td>
                <td className="tabular" style={{ padding: '10px 14px', textAlign: 'right', color: 'var(--green)', fontSize: 11 }}>{pos.tp.toFixed(2)}</td>
                <td className="tabular" style={{ padding: '10px 14px', textAlign: 'right', color: pos.pnl >= 0 ? 'var(--green)' : 'var(--red)', fontWeight: 600 }}>
                  {pos.pnl >= 0 ? '+' : ''}{pos.pnl.toFixed(2)}
                </td>
                <td className="tabular" style={{ padding: '10px 14px', textAlign: 'right' }}>
                  <span style={{ color: pos.bars_held >= horizon - 1 ? 'var(--yellow)' : 'var(--text-secondary)' }}>
                    {pos.bars_held}/{horizon}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
