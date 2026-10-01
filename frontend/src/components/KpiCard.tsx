// src/components/KpiCard.tsx
// Large KPI card with flash animation on value change

import { useEffect, useRef, useState } from 'react';

interface KpiCardProps {
  id: string;
  label: string;
  value: number | string;
  format?: (v: number) => string;
  prefix?: string;
  suffix?: string;
  color?: 'green' | 'red' | 'neutral';
  delta?: number;
  small?: boolean;
}

function formatDefault(v: number): string {
  return v.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

export function KpiCard({ id, label, value, format, prefix = '', suffix = '', color, delta, small }: KpiCardProps) {
  const prevValue = useRef<number | string>(value);
  const [flashClass, setFlashClass] = useState('');

  useEffect(() => {
    if (prevValue.current !== value) {
      const prev = Number(prevValue.current);
      const curr = Number(value);
      if (!isNaN(prev) && !isNaN(curr)) {
        const cls = curr >= prev ? 'flash-green' : 'flash-red';
        setFlashClass(cls);
        const t = setTimeout(() => setFlashClass(''), 700);
        prevValue.current = value;
        return () => clearTimeout(t);
      }
      prevValue.current = value;
    }
  }, [value]);

  const displayValue = typeof value === 'number'
    ? (format ? format(value) : formatDefault(value))
    : value;

  const autoColor = color ?? (typeof value === 'number' && value > 0 ? 'green' : typeof value === 'number' && value < 0 ? 'red' : 'neutral');
  const colorVar = autoColor === 'green' ? 'var(--green)' : autoColor === 'red' ? 'var(--red)' : 'var(--text-primary)';

  return (
    <div id={id} className={`card ${flashClass}`} style={{ position: 'relative', overflow: 'hidden' }}>
      {/* Subtle glow strip */}
      <div style={{
        position: 'absolute', top: 0, left: 0, right: 0, height: '2px',
        background: autoColor === 'green' ? 'var(--green)' : autoColor === 'red' ? 'var(--red)' : 'var(--border)',
        opacity: 0.7,
      }} />

      <div style={{ fontSize: 11, color: 'var(--text-muted)', fontWeight: 600, letterSpacing: '0.8px', textTransform: 'uppercase', marginBottom: 8 }}>
        {label}
      </div>

      <div className="tabular" style={{
        fontSize: small ? 22 : 28,
        fontWeight: 700,
        color: colorVar,
        lineHeight: 1.1,
        letterSpacing: '-0.5px',
      }}>
        {prefix}{displayValue}{suffix}
      </div>

      {delta !== undefined && (
        <div style={{
          marginTop: 6, fontSize: 12, fontWeight: 500,
          color: delta >= 0 ? 'var(--green)' : 'var(--red)',
          display: 'flex', alignItems: 'center', gap: 4,
        }}>
          {delta >= 0 ? '▲' : '▼'} {Math.abs(delta).toFixed(2)}%
        </div>
      )}
    </div>
  );
}
