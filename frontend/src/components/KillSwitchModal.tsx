// src/components/KillSwitchModal.tsx
// Confirmation modal: user must type "KILL" before the switch fires

import { useState } from 'react';

interface Props {
  onConfirm: () => void;
  onCancel: () => void;
}

export function KillSwitchModal({ onConfirm, onCancel }: Props) {
  const [text, setText] = useState('');
  const ready = text.trim() === 'KILL';

  return (
    <div style={{
      position: 'fixed', inset: 0, zIndex: 9999,
      background: 'rgba(0,0,0,0.8)', backdropFilter: 'blur(4px)',
      display: 'flex', alignItems: 'center', justifyContent: 'center',
    }}>
      <div style={{
        background: 'var(--bg-card)', border: '1px solid rgba(248,113,113,0.4)',
        borderRadius: 16, padding: 32, maxWidth: 420, width: '90%',
        boxShadow: '0 0 60px rgba(248,113,113,0.15)',
      }}>
        <div style={{ fontSize: 24, marginBottom: 8 }}>⚠️ Kill Switch</div>
        <p style={{ color: 'var(--text-secondary)', fontSize: 14, marginBottom: 20 }}>
          This will <strong style={{ color: 'var(--red)' }}>immediately close all bot positions</strong> and stop trading.
          Type <code style={{ background: 'rgba(255,255,255,0.08)', padding: '1px 6px', borderRadius: 4 }}>KILL</code> to confirm.
        </p>

        <input
          id="kill-switch-input"
          type="text"
          value={text}
          onChange={e => setText(e.target.value)}
          placeholder="Type KILL to confirm"
          autoFocus
          style={{
            width: '100%', padding: '10px 14px', borderRadius: 8,
            background: 'rgba(255,255,255,0.06)',
            border: `1px solid ${ready ? 'var(--red)' : 'var(--border)'}`,
            color: ready ? 'var(--red)' : 'var(--text-primary)',
            fontSize: 16, fontWeight: 700, fontFamily: 'JetBrains Mono, monospace',
            outline: 'none', marginBottom: 20,
          }}
        />

        <div style={{ display: 'flex', gap: 12 }}>
          <button
            id="btn-kill-confirm"
            onClick={onConfirm}
            disabled={!ready}
            style={{
              flex: 1, padding: '10px', borderRadius: 8,
              background: ready ? 'var(--red)' : 'rgba(255,255,255,0.05)',
              color: ready ? '#fff' : 'var(--text-muted)',
              border: 'none', cursor: ready ? 'pointer' : 'not-allowed',
              fontWeight: 700, fontSize: 14, transition: 'all 0.2s',
            }}>
            CONFIRM KILL
          </button>
          <button
            id="btn-kill-cancel"
            onClick={onCancel}
            style={{
              padding: '10px 20px', borderRadius: 8,
              background: 'transparent', color: 'var(--text-secondary)',
              border: '1px solid var(--border)', cursor: 'pointer', fontSize: 14,
            }}>
            Cancel
          </button>
        </div>
      </div>
    </div>
  );
}
