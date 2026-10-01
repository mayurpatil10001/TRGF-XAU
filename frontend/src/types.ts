// src/types.ts — TypeScript types matching the backend Pydantic schema

export interface AccountInfo {
  login: number;
  server: string;
  mode: 'DEMO' | 'LIVE';
  currency: string;
}

export interface OpenPosition {
  ticket: number;
  side: 'BUY' | 'SELL';
  lots: number;
  entry: number;
  price: number;
  sl: number;
  tp: number;
  pnl: number;
  bars_held: number;
}

export interface MarketSnapshot {
  bid: number;
  ask: number;
  spread: number;
}

export interface LastSignal {
  bar_time: string;
  action: 'BUY' | 'SELL' | 'NO_TRADE';
  p_long: number;
  p_short: number;
  p_none: number;
  threshold: number;
  reason: string;
}

export interface BotStatus {
  status: 'RUNNING' | 'PAUSED' | 'SIGNAL_ONLY' | 'KILLED';
  daily_loss_used_pct: number;
  consecutive_losses: number;
}

export interface ProfitGateStatus {
  status: 'PASSED' | 'PROVISIONAL' | 'FAILED' | 'UNKNOWN';
  updated: string;
}

export interface LiveSnapshot {
  ts: string;
  seq: number;
  mt5_connected: boolean;
  account: AccountInfo;
  balance: number;
  equity: number;
  margin: number;
  free_margin: number;
  floating_pnl: number;
  realized_pnl_today: number;
  realized_pnl_total: number;
  open_positions: OpenPosition[];
  market: MarketSnapshot;
  last_signal: LastSignal;
  bot: BotStatus;
  profit_gate: ProfitGateStatus;
}
