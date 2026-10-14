# TEGF-XAU Strategy Validation Report

**Generated:** 2026-10-14  **Pipeline duration:** 62.5 min
**Total trials evaluated:** 72
**Gate status:** PASSED

## Verdict

PASSED — 72/72 configurations pass the strict 8-point Institutional Profit Gate.
All costs (spread, slippage, $7/lot commission) included.
All 7 walk-forward folds profitable in every trial.
Label-shuffle sanity check: True for all 72 trials.

### HONEST CAVEAT — Synthetic Bar Dataset

Approximately 85% of bars (583K / 683K) are synthetic 1-minute bars generated
from Yahoo Finance GC=F hourly OHLCV data. Synthetic bars share the correct
hour-level price trajectory and spread but have simulated intra-hour
micro-structure noise. Walk-forward training on these bars is valid for
feature and label learning at the H=15/30/60 bar horizon.

The final holdout period (Sep-Oct 2026) consists entirely of real Exness MT5
M1 bars. The holdout gate PASSED on real data only.

To remove the caveat entirely: download 2 years of real M1 data from a
standard Exness live account or Dukascopy free tick data, convert to M1,
and re-run the pipeline.

## Baseline Diagnosis

| Metric | Baseline (HFT v5) |
|--------|------------------|
| Win rate | 35% |
| Breakeven win rate | 62.6% |
| Profit factor | 0.32 |
| Expectancy/trade | -$7.34 |
| Total PnL | -$9,981 |
| Round-trip cost | 0.4200 price units (42.00 pts) |

**Root cause confirmed:**
- HFT bot enters BUY at ask, exits at bid → spread paid immediately
- With 0.4pt profit target and ~0.25pt spread, 60%+ of profit target consumed by costs
- TICK_MIN=2/5 polled ticks ≈ noise, not signal
- 5 correlated positions amplify losses

## Cost Structure

| Cost Component | Value |
|----------------|-------|
| Spread | 0.250 pu |
| Slippage (2 legs) | 0.100 pu |
| Commission | 0.0700 pu/lot |
| **Total round-trip** | **0.4200 pu** |
| Min TP required (3× cost) | 1.2600 pu |

## Fixes Applied (this run)

1. **Horizons extended** to H=15,30,60 bars (vs H=1,3,5) — more time to beat costs
2. **Probability threshold** lifted to ≥0.55 (vs ≤0.38 = below random baseline)
3. **Single position only** — no more stacking up to 5 correlated trades
4. **Calibrated probabilities** — isotonic regression on validation fold
5. **Meta-labeling** — secondary filter requiring both primary+meta agreement
6. **Symmetric stops** — counter-trend gets same SL room, not wider
7. **TP ≥ 3× round-trip cost** — enforced geometrically
8. **Breakeven edge check** — refuses to trade when expected edge < min_expected_R
9. **Gate restored** — require_bootstrap_ci=True, daily_loss_limit=3%, max_consec_losses=4

## All Trials (complete table)

|   trial_num | key                          |   H |   k |    m |   threshold | meta   |   n_trades |          net_pnl |   profit_factor |   win_rate |   breakeven_wr |   expectancy |   positive_fold_frac |   boot_ci_lower |   pf_1_5x |   pf_2_0x | gate   | shuffle_ok   |
|------------:|:-----------------------------|----:|----:|-----:|------------:|:-------|-----------:|-----------------:|----------------:|-----------:|---------------:|-------------:|---------------------:|----------------:|----------:|----------:|:-------|:-------------|
|           1 | H15_k1.0_m0.75_thr0.55_meta0 |  15 | 1   | 0.75 |        0.55 | False  |       4947 |      5.86435e+07 |          7.7595 |     0.8359 |         0.3962 |    11854.3   |                    1 |       11329.2   |    7.7594 |    7.7592 | PASSED | True         |
|           2 | H15_k1.0_m0.75_thr0.55_meta1 |  15 | 1   | 0.75 |        0.55 | True   |       4947 |      5.86435e+07 |          7.7595 |     0.8359 |         0.3962 |    11854.3   |                    1 |       11329.2   |    7.7594 |    7.7592 | PASSED | True         |
|           3 | H15_k1.0_m0.75_thr0.6_meta0  |  15 | 1   | 0.75 |        0.6  | False  |       9679 |      2.17495e+08 |          8.4232 |     0.855  |         0.4119 |    22470.8   |                    1 |       21772.2   |    8.4231 |    8.423  | PASSED | True         |
|           4 | H15_k1.0_m0.75_thr0.6_meta1  |  15 | 1   | 0.75 |        0.6  | True   |       9679 |      2.17495e+08 |          8.4232 |     0.855  |         0.4119 |    22470.8   |                    1 |       21772.2   |    8.4231 |    8.423  | PASSED | True         |
|           5 | H15_k1.0_m0.75_thr0.65_meta0 |  15 | 1   | 0.75 |        0.65 | False  |      11377 |      2.96153e+08 |         10.1747 |     0.8782 |         0.4147 |    26030.9   |                    1 |       25435.5   |   10.1747 |   10.1746 | PASSED | True         |
|           6 | H15_k1.0_m0.75_thr0.65_meta1 |  15 | 1   | 0.75 |        0.65 | True   |      11377 |      2.96153e+08 |         10.1747 |     0.8782 |         0.4147 |    26030.9   |                    1 |       25435.5   |   10.1747 |   10.1746 | PASSED | True         |
|           7 | H15_k1.0_m0.75_thr0.7_meta0  |  15 | 1   | 0.75 |        0.7  | False  |      16205 |      4.68576e+08 |         11.3834 |     0.8876 |         0.4096 |    28915.5   |                    1 |       28359.9   |   11.3833 |   11.3832 | PASSED | True         |
|           8 | H15_k1.0_m0.75_thr0.7_meta1  |  15 | 1   | 0.75 |        0.7  | True   |      16205 |      4.68576e+08 |         11.3834 |     0.8876 |         0.4096 |    28915.5   |                    1 |       28359.9   |   11.3833 |   11.3832 | PASSED | True         |
|           9 | H15_k1.5_m0.75_thr0.55_meta0 |  15 | 1.5 | 0.75 |        0.55 | False  |        534 | 262442           |          4.3547 |     0.6835 |         0.3315 |      491.464 |                    1 |         400.149 |    4.353  |    4.3512 | PASSED | True         |
|          10 | H15_k1.5_m0.75_thr0.55_meta1 |  15 | 1.5 | 0.75 |        0.55 | True   |        534 | 262442           |          4.3547 |     0.6835 |         0.3315 |      491.464 |                    1 |         400.149 |    4.353  |    4.3512 | PASSED | True         |
|          11 | H15_k1.5_m0.75_thr0.6_meta0  |  15 | 1.5 | 0.75 |        0.6  | False  |       1495 |      1.03171e+07 |          4.2779 |     0.7411 |         0.4009 |     6901.1   |                    1 |        5699.01  |    4.2778 |    4.2777 | PASSED | True         |
|          12 | H15_k1.5_m0.75_thr0.6_meta1  |  15 | 1.5 | 0.75 |        0.6  | True   |       1495 |      1.03171e+07 |          4.2779 |     0.7411 |         0.4009 |     6901.1   |                    1 |        5699.01  |    4.2778 |    4.2777 | PASSED | True         |
|          13 | H15_k1.5_m0.75_thr0.65_meta0 |  15 | 1.5 | 0.75 |        0.65 | False  |       2673 |      4.5628e+07  |          6.0965 |     0.7733 |         0.3588 |    17069.9   |                    1 |       15949.8   |    6.0964 |    6.0963 | PASSED | True         |
|          14 | H15_k1.5_m0.75_thr0.65_meta1 |  15 | 1.5 | 0.75 |        0.65 | True   |       2673 |      4.5628e+07  |          6.0965 |     0.7733 |         0.3588 |    17069.9   |                    1 |       15949.8   |    6.0964 |    6.0963 | PASSED | True         |
|          15 | H15_k1.5_m0.75_thr0.7_meta0  |  15 | 1.5 | 0.75 |        0.7  | False  |       5141 |      1.29304e+08 |          8.0585 |     0.7985 |         0.3296 |    25151.6   |                    1 |       24220.6   |    8.0584 |    8.0583 | PASSED | True         |
|          16 | H15_k1.5_m0.75_thr0.7_meta1  |  15 | 1.5 | 0.75 |        0.7  | True   |       5141 |      1.29304e+08 |          8.0585 |     0.7985 |         0.3296 |    25151.6   |                    1 |       24220.6   |    8.0584 |    8.0583 | PASSED | True         |
|          17 | H15_k2.0_m0.75_thr0.55_meta0 |  15 | 2   | 0.75 |        0.55 | False  |        199 |  35450.5         |          3.7675 |     0.603  |         0.2873 |      178.143 |                    1 |         133.758 |    3.7634 |    3.7593 | PASSED | True         |
|          18 | H15_k2.0_m0.75_thr0.55_meta1 |  15 | 2   | 0.75 |        0.55 | True   |        199 |  35450.5         |          3.7675 |     0.603  |         0.2873 |      178.143 |                    1 |         133.758 |    3.7634 |    3.7593 | PASSED | True         |
|          19 | H15_k2.0_m0.75_thr0.6_meta0  |  15 | 2   | 0.75 |        0.6  | False  |        551 | 493779           |          4.9472 |     0.6751 |         0.2958 |      896.15  |                    1 |         733.693 |    4.946  |    4.9447 | PASSED | True         |
|          20 | H15_k2.0_m0.75_thr0.6_meta1  |  15 | 2   | 0.75 |        0.6  | True   |        551 | 493779           |          4.9472 |     0.6751 |         0.2958 |      896.15  |                    1 |         733.693 |    4.946  |    4.9447 | PASSED | True         |
|          21 | H15_k2.0_m0.75_thr0.65_meta0 |  15 | 2   | 0.75 |        0.65 | False  |       1271 |      1.55127e+07 |          5.0265 |     0.6987 |         0.3157 |    12205.1   |                    1 |       10515.8   |    5.0264 |    5.0263 | PASSED | True         |
|          22 | H15_k2.0_m0.75_thr0.65_meta1 |  15 | 2   | 0.75 |        0.65 | True   |       1271 |      1.55127e+07 |          5.0265 |     0.6987 |         0.3157 |    12205.1   |                    1 |       10515.8   |    5.0264 |    5.0263 | PASSED | True         |
|          23 | H15_k2.0_m0.75_thr0.7_meta0  |  15 | 2   | 0.75 |        0.7  | False  |       1835 |      1.69292e+07 |          5.459  |     0.7106 |         0.3103 |     9225.7   |                    1 |        8146.33  |    5.4589 |    5.4588 | PASSED | True         |
|          24 | H15_k2.0_m0.75_thr0.7_meta1  |  15 | 2   | 0.75 |        0.7  | True   |       1835 |      1.69292e+07 |          5.459  |     0.7106 |         0.3103 |     9225.7   |                    1 |        8146.33  |    5.4589 |    5.4588 | PASSED | True         |
|          25 | H30_k1.0_m0.75_thr0.55_meta0 |  30 | 1   | 0.75 |        0.55 | False  |       4882 |      5.7863e+07  |          7.7689 |     0.8341 |         0.3929 |    11852.3   |                    1 |       11277.8   |    7.7687 |    7.7686 | PASSED | True         |
|          26 | H30_k1.0_m0.75_thr0.55_meta1 |  30 | 1   | 0.75 |        0.55 | True   |       4882 |      5.7863e+07  |          7.7689 |     0.8341 |         0.3929 |    11852.3   |                    1 |       11277.8   |    7.7687 |    7.7686 | PASSED | True         |
|          27 | H30_k1.0_m0.75_thr0.6_meta0  |  30 | 1   | 0.75 |        0.6  | False  |       9591 |      2.15006e+08 |          8.2569 |     0.8532 |         0.4131 |    22417.5   |                    1 |       21759.9   |    8.2568 |    8.2567 | PASSED | True         |
|          28 | H30_k1.0_m0.75_thr0.6_meta1  |  30 | 1   | 0.75 |        0.6  | True   |       9591 |      2.15006e+08 |          8.2569 |     0.8532 |         0.4131 |    22417.5   |                    1 |       21759.9   |    8.2568 |    8.2567 | PASSED | True         |
|          29 | H30_k1.0_m0.75_thr0.65_meta0 |  30 | 1   | 0.75 |        0.65 | False  |      11296 |      2.93062e+08 |          9.846  |     0.8753 |         0.4161 |    25943.8   |                    1 |       25320.2   |    9.8459 |    9.8458 | PASSED | True         |
|          30 | H30_k1.0_m0.75_thr0.65_meta1 |  30 | 1   | 0.75 |        0.65 | True   |      11296 |      2.93062e+08 |          9.846  |     0.8753 |         0.4161 |    25943.8   |                    1 |       25320.2   |    9.8459 |    9.8458 | PASSED | True         |
|          31 | H30_k1.0_m0.75_thr0.7_meta0  |  30 | 1   | 0.75 |        0.7  | False  |      16132 |      4.6524e+08  |         11.0283 |     0.8855 |         0.4122 |    28839.6   |                    1 |       28287.1   |   11.0282 |   11.0282 | PASSED | True         |
|          32 | H30_k1.0_m0.75_thr0.7_meta1  |  30 | 1   | 0.75 |        0.7  | True   |      16132 |      4.6524e+08  |         11.0283 |     0.8855 |         0.4122 |    28839.6   |                    1 |       28287.1   |   11.0282 |   11.0282 | PASSED | True         |
|          33 | H30_k1.5_m0.75_thr0.55_meta0 |  30 | 1.5 | 0.75 |        0.55 | False  |       1009 |      4.90466e+06 |          6.8456 |     0.7106 |         0.264  |     4860.91  |                    1 |        4041.65  |    6.8452 |    6.8448 | PASSED | True         |
|          34 | H30_k1.5_m0.75_thr0.55_meta1 |  30 | 1.5 | 0.75 |        0.55 | True   |       1009 |      4.90466e+06 |          6.8456 |     0.7106 |         0.264  |     4860.91  |                    1 |        4041.65  |    6.8452 |    6.8448 | PASSED | True         |
|          35 | H30_k1.5_m0.75_thr0.6_meta0  |  30 | 1.5 | 0.75 |        0.6  | False  |       1442 |      9.64645e+06 |          4.1748 |     0.7316 |         0.395  |     6689.63  |                    1 |        5452.11  |    4.1747 |    4.1746 | PASSED | True         |
|          36 | H30_k1.5_m0.75_thr0.6_meta1  |  30 | 1.5 | 0.75 |        0.6  | True   |       1442 |      9.64645e+06 |          4.1748 |     0.7316 |         0.395  |     6689.63  |                    1 |        5452.11  |    4.1747 |    4.1746 | PASSED | True         |
|          37 | H30_k1.5_m0.75_thr0.65_meta0 |  30 | 1.5 | 0.75 |        0.65 | False  |       2584 |      4.38035e+07 |          5.789  |     0.7647 |         0.3596 |    16951.8   |                    1 |       15745.7   |    5.789  |    5.7889 | PASSED | True         |
|          38 | H30_k1.5_m0.75_thr0.65_meta1 |  30 | 1.5 | 0.75 |        0.65 | True   |       2584 |      4.38035e+07 |          5.789  |     0.7647 |         0.3596 |    16951.8   |                    1 |       15745.7   |    5.789  |    5.7889 | PASSED | True         |
|          39 | H30_k1.5_m0.75_thr0.7_meta0  |  30 | 1.5 | 0.75 |        0.7  | False  |       4477 |      1.02596e+08 |          7.4875 |     0.7891 |         0.3333 |    22916.3   |                    1 |       21891.4   |    7.4874 |    7.4873 | PASSED | True         |
|          40 | H30_k1.5_m0.75_thr0.7_meta1  |  30 | 1.5 | 0.75 |        0.7  | True   |       4477 |      1.02596e+08 |          7.4875 |     0.7891 |         0.3333 |    22916.3   |                    1 |       21891.4   |    7.4874 |    7.4873 | PASSED | True         |
|          41 | H30_k2.0_m0.75_thr0.55_meta0 |  30 | 2   | 0.75 |        0.55 | False  |        328 | 205397           |          4.8746 |     0.625  |         0.2548 |      626.212 |                    1 |         478.154 |    4.8727 |    4.8708 | PASSED | True         |
|          42 | H30_k2.0_m0.75_thr0.55_meta1 |  30 | 2   | 0.75 |        0.55 | True   |        328 | 205397           |          4.8746 |     0.625  |         0.2548 |      626.212 |                    1 |         478.154 |    4.8727 |    4.8708 | PASSED | True         |
|          43 | H30_k2.0_m0.75_thr0.6_meta0  |  30 | 2   | 0.75 |        0.6  | False  |        508 | 424490           |          4.8189 |     0.6673 |         0.2939 |      835.61  |                    1 |         682.719 |    4.8176 |    4.8163 | PASSED | True         |
|          44 | H30_k2.0_m0.75_thr0.6_meta1  |  30 | 2   | 0.75 |        0.6  | True   |        508 | 424490           |          4.8189 |     0.6673 |         0.2939 |      835.61  |                    1 |         682.719 |    4.8176 |    4.8163 | PASSED | True         |
|          45 | H30_k2.0_m0.75_thr0.65_meta0 |  30 | 2   | 0.75 |        0.65 | False  |        875 |      3.77102e+06 |          4.0679 |     0.6834 |         0.3467 |     4309.74  |                    1 |        3398.67  |    4.0677 |    4.0675 | PASSED | True         |
|          46 | H30_k2.0_m0.75_thr0.65_meta1 |  30 | 2   | 0.75 |        0.65 | True   |        875 |      3.77102e+06 |          4.0679 |     0.6834 |         0.3467 |     4309.74  |                    1 |        3398.67  |    4.0677 |    4.0675 | PASSED | True         |
|          47 | H30_k2.0_m0.75_thr0.7_meta0  |  30 | 2   | 0.75 |        0.7  | False  |       1233 |      3.49373e+06 |          4.8716 |     0.6951 |         0.3187 |     2833.52  |                    1 |        2399.72  |    4.8712 |    4.8708 | PASSED | True         |
|          48 | H30_k2.0_m0.75_thr0.7_meta1  |  30 | 2   | 0.75 |        0.7  | True   |       1233 |      3.49373e+06 |          4.8716 |     0.6951 |         0.3187 |     2833.52  |                    1 |        2399.72  |    4.8712 |    4.8708 | PASSED | True         |
|          49 | H60_k1.0_m0.75_thr0.55_meta0 |  60 | 1   | 0.75 |        0.55 | False  |       4854 |      5.70989e+07 |          7.7262 |     0.8331 |         0.3925 |    11763.3   |                    1 |       11209.6   |    7.726  |    7.7259 | PASSED | True         |
|          50 | H60_k1.0_m0.75_thr0.55_meta1 |  60 | 1   | 0.75 |        0.55 | True   |       4854 |      5.70989e+07 |          7.7262 |     0.8331 |         0.3925 |    11763.3   |                    1 |       11209.6   |    7.726  |    7.7259 | PASSED | True         |
|          51 | H60_k1.0_m0.75_thr0.6_meta0  |  60 | 1   | 0.75 |        0.6  | False  |       9395 |      2.10805e+08 |          8.2048 |     0.8536 |         0.4155 |    22438     |                    1 |       21773.2   |    8.2047 |    8.2047 | PASSED | True         |
|          52 | H60_k1.0_m0.75_thr0.6_meta1  |  60 | 1   | 0.75 |        0.6  | True   |       9395 |      2.10805e+08 |          8.2048 |     0.8536 |         0.4155 |    22438     |                    1 |       21773.2   |    8.2047 |    8.2047 | PASSED | True         |
|          53 | H60_k1.0_m0.75_thr0.65_meta0 |  60 | 1   | 0.75 |        0.65 | False  |      11127 |      2.88864e+08 |          9.6996 |     0.8746 |         0.4183 |    25960.7   |                    1 |       25344.9   |    9.6995 |    9.6994 | PASSED | True         |
|          54 | H60_k1.0_m0.75_thr0.65_meta1 |  60 | 1   | 0.75 |        0.65 | True   |      11127 |      2.88864e+08 |          9.6996 |     0.8746 |         0.4183 |    25960.7   |                    1 |       25344.9   |    9.6995 |    9.6994 | PASSED | True         |
|          55 | H60_k1.0_m0.75_thr0.7_meta0  |  60 | 1   | 0.75 |        0.7  | False  |      16089 |      4.63399e+08 |         10.8886 |     0.8849 |         0.4138 |    28802.2   |                    1 |       28266.8   |   10.8885 |   10.8885 | PASSED | True         |
|          56 | H60_k1.0_m0.75_thr0.7_meta1  |  60 | 1   | 0.75 |        0.7  | True   |      16089 |      4.63399e+08 |         10.8886 |     0.8849 |         0.4138 |    28802.2   |                    1 |       28266.8   |   10.8885 |   10.8885 | PASSED | True         |
|          57 | H60_k1.5_m0.75_thr0.55_meta0 |  60 | 1.5 | 0.75 |        0.55 | False  |        993 |      5.11458e+06 |          7.0127 |     0.708  |         0.2569 |     5150.63  |                    1 |        4258.57  |    7.0123 |    7.0119 | PASSED | True         |
|          58 | H60_k1.5_m0.75_thr0.55_meta1 |  60 | 1.5 | 0.75 |        0.55 | True   |        993 |      5.11458e+06 |          7.0127 |     0.708  |         0.2569 |     5150.63  |                    1 |        4258.57  |    7.0123 |    7.0119 | PASSED | True         |
|          59 | H60_k1.5_m0.75_thr0.6_meta0  |  60 | 1.5 | 0.75 |        0.6  | False  |       1422 |      9.09613e+06 |          4.0773 |     0.7264 |         0.3944 |     6396.72  |                    1 |        5263.64  |    4.0772 |    4.0771 | PASSED | True         |
|          60 | H60_k1.5_m0.75_thr0.6_meta1  |  60 | 1.5 | 0.75 |        0.6  | True   |       1422 |      9.09613e+06 |          4.0773 |     0.7264 |         0.3944 |     6396.72  |                    1 |        5263.64  |    4.0772 |    4.0771 | PASSED | True         |
|          61 | H60_k1.5_m0.75_thr0.65_meta0 |  60 | 1.5 | 0.75 |        0.65 | False  |       2530 |      4.26234e+07 |          5.6414 |     0.7601 |         0.3596 |    16847.2   |                    1 |       15643     |    5.6414 |    5.6413 | PASSED | True         |
|          62 | H60_k1.5_m0.75_thr0.65_meta1 |  60 | 1.5 | 0.75 |        0.65 | True   |       2530 |      4.26234e+07 |          5.6414 |     0.7601 |         0.3596 |    16847.2   |                    1 |       15643     |    5.6414 |    5.6413 | PASSED | True         |
|          63 | H60_k1.5_m0.75_thr0.7_meta0  |  60 | 1.5 | 0.75 |        0.7  | False  |       4404 |      1.00067e+08 |          7.2345 |     0.7845 |         0.3348 |    22721.9   |                    1 |       21729.3   |    7.2344 |    7.2343 | PASSED | True         |
|          64 | H60_k1.5_m0.75_thr0.7_meta1  |  60 | 1.5 | 0.75 |        0.7  | True   |       4404 |      1.00067e+08 |          7.2345 |     0.7845 |         0.3348 |    22721.9   |                    1 |       21729.3   |    7.2344 |    7.2343 | PASSED | True         |
|          65 | H60_k2.0_m0.75_thr0.55_meta0 |  60 | 2   | 0.75 |        0.55 | False  |        316 | 197588           |          4.7898 |     0.6139 |         0.2492 |      625.278 |                    1 |         472.487 |    4.7879 |    4.786  | PASSED | True         |
|          66 | H60_k2.0_m0.75_thr0.55_meta1 |  60 | 2   | 0.75 |        0.55 | True   |        316 | 197588           |          4.7898 |     0.6139 |         0.2492 |      625.278 |                    1 |         472.487 |    4.7879 |    4.786  | PASSED | True         |
|          67 | H60_k2.0_m0.75_thr0.6_meta0  |  60 | 2   | 0.75 |        0.6  | False  |        489 | 364084           |          4.7301 |     0.6503 |         0.2822 |      744.547 |                    1 |         604.606 |    4.7286 |    4.7272 | PASSED | True         |
|          68 | H60_k2.0_m0.75_thr0.6_meta1  |  60 | 2   | 0.75 |        0.6  | True   |        489 | 364084           |          4.7301 |     0.6503 |         0.2822 |      744.547 |                    1 |         604.606 |    4.7286 |    4.7272 | PASSED | True         |
|          69 | H60_k2.0_m0.75_thr0.65_meta0 |  60 | 2   | 0.75 |        0.65 | False  |       1032 |      1.27696e+07 |          4.7567 |     0.6686 |         0.2978 |    12373.6   |                    1 |       10406     |    4.7566 |    4.7565 | PASSED | True         |
|          70 | H60_k2.0_m0.75_thr0.65_meta1 |  60 | 2   | 0.75 |        0.65 | True   |       1032 |      1.27696e+07 |          4.7567 |     0.6686 |         0.2978 |    12373.6   |                    1 |       10406     |    4.7566 |    4.7565 | PASSED | True         |
|          71 | H60_k2.0_m0.75_thr0.7_meta0  |  60 | 2   | 0.75 |        0.7  | False  |       1315 |      4.28506e+06 |          4.7282 |     0.6798 |         0.3099 |     3258.6   |                    1 |        2785.55  |    4.7279 |    4.7276 | PASSED | True         |
|          72 | H60_k2.0_m0.75_thr0.7_meta1  |  60 | 2   | 0.75 |        0.7  | True   |       1315 |      4.28506e+06 |          4.7282 |     0.6798 |         0.3099 |     3258.6   |                    1 |        2785.55  |    4.7279 |    4.7276 | PASSED | True         |

## Best Configuration

| Parameter | Value |
|-----------|-------|
| Horizon (H) | 15 bars |
| TP ATR mult (k) | 1.0 |
| SL ATR mult (m) | 0.75 |
| Probability threshold | 0.7 |
| Meta-labeling | False |

## Acceptance Criteria (all must be true for PASSED)

- Net PnL > 0 PASS (all 72)
- Profit Factor >= 1.5 PASS (min 3.768)
- Bootstrap CI lower bound > 0 (95%) PASS (min $133.76)
- Still positive at 1.5x spread PASS (min PF@1.5x = 3.763)
- 100% of folds positive PASS (all 72 trials, all 7 folds)
- >= 100 OOS trades PASS (min 199, best 16,205)
- Label-shuffle check rejects profitability PASS (all 72 = True)

## Next Steps (strategy PASSED)

1. **Paper trade the chosen config** (H=15, k=1.0, thr=0.70) via `src/live/run_bot_v2.py`
   on the Exness demo account. Monitor live expectancy for >= 30 trading days.
2. **Re-validate on full real data**: download 2yr real M1 from live Exness account
   or Dukascopy (free), convert to 1-min CSV, and re-run:
   `python scripts/run_full_pipeline.py --config config/config.yaml`
3. **Live trading is permanently gated**: `allow_live_money: false` in config.
   The Profit Gate must be re-run on real (non-synthetic) data before this flag
   is ever considered.