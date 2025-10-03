# Finding Regime Periods

# A) Fundamental data to collect (both countries in the pair)

1. **Rates & expectations**

   * Policy rate, meeting dates/decisions, guidance.
   * OIS/SONIA/TONA/FF futures-implied path (1m–24m), 2y & 10y gov’t yields, curve slopes.
   * **Rate differentials** (home − foreign) for policy, 2y, 10y, slope.
2. **Inflation & growth**

   * CPI headline/core YoY, PPI, inflation swaps (e.g., 5y5y if available).
   * PMIs (mfg/services), payrolls/unemployment, GDP nowcasts.
3. **Surprise/flow proxies**

   * Economic surprise indices (or compute surprises = actual − median forecast).
   * Balance of payments (portfolio flows), FX intervention announcements.
4. **Risk & cross-asset**

   * VIX/MOVE, credit spreads (CDX/ITRAXX), key commodities tied to the bloc (oil for JPY is risk proxy; for CAD/AUD/NOK use terms-of-trade commodities).
5. **CB balance sheets & QE/QT**

   * Total assets (BoJ, BoE, etc.), announcement dates.
6. **Event flags**

   * Referendum/election dates, war/shock dates, sanctions, trade-war phases.

# B) Procedures to carve regimes

## 1) Preprocess & align

* Align everything to a common calendar (daily best; forward-fill to your bar granularity).
* Build **spreads/differentials** (e.g., UK–JP 2y, CPI, PMI).
* Create **surprise series**: `(actual − consensus)/rolling σ` (z-score).
* Compute **market-state features** on the FX itself: realized vol (e.g., 20/60d), return skew/kurt, carry (rate_diff), trend strength (rolling Sharpe of returns).

## 2) Exogenous (event-driven) boundaries

* Create **hard regime breaks** around:

  * ≥25–50 bp hikes/cuts; start/end of QE/QT; YCC changes; major guidance pivots.
  * Macro shocks: COVID crash start (2020-02/03), invasion (2022-02), Brexit vote (2016-06), etc.
* Use **event windows** (e.g., ±10 trading days; or “post window” until next major event).
* Keep these as “anchor regimes” (clean, interpretable).

## 3) Endogenous (data-driven) segmentation

Run one or more of the following on *drivers* and *FX features*:

* **Change-point detection** on:

  * Rate differential, inflation differential, yield curve slope differential, FX realized vol.
  * Methods: Binary Segmentation / PELT (cost = mean/variance shift), CUSUM.
  * Heuristics: penalize many breaks, enforce **min regime length** (e.g., ≥ 60 trading days).
* **Hidden Markov Model (HMM)** with inputs like
  `X_t = [FX ret, FX realized vol, rate_diff, VIX]` → 2–3 states (risk-on carry vs risk-off vs chop).

  * Use smoothed state probabilities; assign regime by argmax; apply **hysteresis** (need k days before switching).
* **MS-VAR / Markov-switching AR** on FX returns with exogenous rate_diff/vol factors if you want more structure.

> Practical tip: do **change-point** on macro *differentials* + **HMM** on price/vol features; then intersect/union them with the **event anchors** to finalize regimes.

## 4) Consolidate regimes

* Merge: start with **event anchors**, then **refine inside each anchor** with change-points/HMM.
* Apply **business rules**:

  * Minimum duration (e.g., ≥ 40–60 bars).
  * Hysteresis (e.g., require 5 consecutive days in a new HMM state before switching).
  * Boundary smoothing (shift boundaries ±5–10 days to avoid spurious flips).

## 5) Validate the cuts

* For each regime:

  * Recompute **distributional stats** of FX returns (μ, σ, skew, vol-of-vol).
  * Check **driver separation** (e.g., Wilcoxon test: are rate_diff means different across regimes?).
  * Run your strategy + **MCPT per regime**. Record PF, p-value, CI.
* **Sensitivity analysis**:

  * Shift each boundary ±k days; does MCPT conclusion hold?
  * Swap one method (HMM → change-point only) and compare.

## 6) Automate & monitor

* Build a pipeline:

  1. Ingest fundamentals → align → engineer features.
  2. Generate event anchors from a calendar file.
  3. Run change-point + HMM.
  4. Merge & clean regimes with rules.
  5. Output labeled periods; run MCPT per label.
* Recompute monthly; store regime labels so you can compare out-of-sample updates.

# What to start with (minimal viable set)

* **For any pair**: policy rate, 2y yield, CPI core YoY (both countries), VIX, FX realized vol, FX returns.
* Engineer: `rate_diff_2y`, `cpi_diff_core`, `rv_20d`.
* Do: PELT on `rate_diff_2y` and `rv_20d`; 2-state HMM on `[ret, rv]`.
* Combine with hard event dates (COVID, 2022 hikes).
* Run MCPT per regime.

# Tooling hints (since you’re in C++/Python)

* **Python** (fast to iterate): `ruptures` (PELT, Binseg), `hmmlearn` or `pomegranate` for HMMs.
* **C++**: implement PELT/CUSUM (simple), or wrap a lightweight HMM (Eigen + EM). You already have bindings; keep the regime labeling in Python first if that’s faster.

# Decision rule to keep it objective

* A regime definition is “good” if within it:

  * FX return distribution is stationary enough (no further strong breaks),
  * Drivers (rate/vol) are statistically distinct from adjacent regimes,
  * Your strategy’s **MCPT p-value** is stable under small boundary shifts.

If you want, I can sketch a concrete pipeline for **GBPJPY** with the exact series names/transformations and the sequence of algorithms (and default thresholds) so you can drop it into your repo.
