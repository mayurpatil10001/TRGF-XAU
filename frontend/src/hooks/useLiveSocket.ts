// src/hooks/useLiveSocket.ts
// WebSocket hook with auto-reconnect, stale detection, and state machine.

import { useState, useEffect, useRef, useCallback } from 'react';
import type { LiveSnapshot } from '../types';

type WsState = 'CONNECTING' | 'LIVE' | 'STALE' | 'RECONNECTING';

interface UseLiveSocketResult {
  snapshot: LiveSnapshot | null;
  wsState: WsState;
  stale: boolean;
}

const DEFAULT_SNAPSHOT: LiveSnapshot | null = null;
const STALE_TIMEOUT_MS = 5000;
const MIN_RECONNECT_MS = 1000;
const MAX_RECONNECT_MS = 15000;

export function useLiveSocket(url: string): UseLiveSocketResult {
  const [snapshot, setSnapshot] = useState<LiveSnapshot | null>(DEFAULT_SNAPSHOT);
  const [wsState, setWsState] = useState<WsState>('CONNECTING');
  const [stale, setStale] = useState(false);

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectDelay = useRef(MIN_RECONNECT_MS);
  const staleTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const resetStaleTimer = useCallback(() => {
    if (staleTimer.current) clearTimeout(staleTimer.current);
    setStale(false);
    staleTimer.current = setTimeout(() => {
      setStale(true);
      setWsState('STALE');
    }, STALE_TIMEOUT_MS);
  }, []);

  const connect = useCallback(() => {
    setWsState('CONNECTING');
    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => {
      setWsState('LIVE');
      reconnectDelay.current = MIN_RECONNECT_MS;
      resetStaleTimer();
    };

    ws.onmessage = (evt) => {
      try {
        const data: LiveSnapshot = JSON.parse(evt.data);
        setSnapshot(data);
        setWsState('LIVE');
        resetStaleTimer();
      } catch (e) {
        console.warn('WS parse error', e);
      }
    };

    ws.onerror = () => {
      setWsState('STALE');
      setStale(true);
    };

    ws.onclose = () => {
      setWsState('RECONNECTING');
      setStale(true);
      if (staleTimer.current) clearTimeout(staleTimer.current);
      reconnectTimer.current = setTimeout(() => {
        reconnectDelay.current = Math.min(reconnectDelay.current * 2, MAX_RECONNECT_MS);
        connect();
      }, reconnectDelay.current);
    };
  }, [url, resetStaleTimer]);

  useEffect(() => {
    connect();
    return () => {
      if (wsRef.current) wsRef.current.close();
      if (staleTimer.current) clearTimeout(staleTimer.current);
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
    };
  }, [connect]);

  return { snapshot, wsState, stale };
}
