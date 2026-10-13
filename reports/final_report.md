# TEGF-XAU Final Report
**Generated**: 2026-10-13T14:06:12.021205+00:00
**Config hash**: `bc9989b65f32dbb5`

---

## 1. Data Quality
See [`reports/data_qc.md`](data_qc.md) for the full QC report.

## 2. Feature Engineering
Features from Groups A–G (returns, bar shape, volume, volatility, timing proxy, session/time, trend/mean-reversion). All features are strictly causal — no lookahead. Scaler fit on training fold only per walk-forward split.

## 3. Model Comparison
See walk-forward fold artifacts in `artifacts/fold_*/`. Comparison table (directional accuracy, NLL, calibration) generated per fold.

## 4. Profit Gate
**Status: FAILED**

| Criterion | Value | Threshold | Pass |
|-----------|-------|-----------|------|
| total_net_profit_positive | -9981.64 | > 0 | ❌ |
| bootstrap_ci_lower_gt_zero | -9.705748 | > 0 (95% CI lower bound) | ✅ |
| profit_factor_gte_threshold | 0.3222 | >= 1.1 | ❌ |
| positive_at_stress_spread | -10151.64 | > 0 at 1.5x spread | ❌ |
| min_oos_trades | 1360 | >= 30 | ✅ |
| positive_fold_fraction | 0.0 | >= 0.6 | ❌ |
| holdout_positive | NOT_EVALUATED | > 0 (or NOT_EVALUATED for dev-phase) | ✅ |
| shuffle_sanity_test | PASSED | MUST PASS | ✅ |

### Diagnosis
- Total round-trip cost = 0.4200 price units. At 1-min horizon, required |return| > 0.4200 just to break even before edge. 1-min moves vs 60-min moves have ~7.7x worse signal-to-cost ratio (sqrt-time scaling).
- Achieved win rate: 0.35
- Breakeven win rate: 0.6256

**Next experiments:**
- Increase horizon H to 5 or more bars
- Tighten probability threshold (e.g., 0.65+)
- Restrict to overlap session (highest liquidity, lower effective spread)
- Meta-labelling: add a second filter model
- Use expected-R threshold to only trade high-conviction signals

## 5. Holdout Result
_Holdout not yet evaluated._

## 6. Spread Stress Table
See `reports/spread_stress.json`.


## Known Limitations

1. **Bar data vs tick data**: Features are computed from 1-minute OHLC. The timing proxy
   (Group E) approximates intra-bar high/low order from bar structure. Agreement rate with
   true tick-level timing is validated in `scripts/compare_timing_true_vs_proxy.py`.

2. **Spread assumptions**: Pre-2015 data uses a synthetic constant spread which may
   underestimate (or overestimate) historical transaction costs. Results before 2015
   should be interpreted as regime-sensitivity analysis only.

3. **Slippage model**: Constant slippage is a simplification. Live fills on gold in
   thin conditions may be significantly higher. Verified in the live-vs-backtest drift panel.

4. **1-minute horizon cost penalty**: The cost-to-signal ratio at 1-minute is ~7.7×
   worse than hourly forecasting (sqrt-time scaling). This is the primary challenge;
   longer holds or highly selective trading are the honest levers for profitability.

5. **Multiple-testing caveat**: The bounded search tried multiple (H, k, m, threshold,
   session) combinations on validation folds. The Deflated Sharpe Ratio (DSR) corrects
   for this. The number of trials is logged in `reports/trials.csv`.

6. **Non-stationarity**: Gold microstructure has changed materially since 2001
   (electronic trading, HFT, ETF growth). Models trained on 2015–2022 may not
   generalize to post-2024 conditions.

7. **Broker differences**: Spread, commission, and execution quality vary by broker.
   All results assume the configured constant spread unless a real spread column exists.

## Honest Conclusion
A rigorous null result — where no reliable net-of-cost edge is found at the 1-minute horizon — is a valid and valuable scientific outcome. The pipeline correctly identifies when costs exceed the signal, and provides a clear diagnosis for where edge, if any, may exist.
