Short answer: safe if you enforce strict causality. Two common leaks to avoid: (1) back-dating regime flips at confirmation and (2) using smoothed HMM or full-sample preprocessing.

# Causality contract

* Compute features with past-only data at bar close.
* Consume indicator values with `.shift(1)` when making next-bar decisions.
* Never revise past indicator rows after they’re written.

# HMM specifics

* Use **forward filter** α_t = p(state | y_1:t).
* Do **not** use smoothed posteriors p(state | y_1:T) or Viterbi paths; they look ahead.
* Train on in-sample only. For backtests, use walk-forward or expanding refits.

# DC/TMV/R specifics

* A DC event is “known” only at or after the bar that closes it.
* Write the event and any regime change **on the confirmation bar**. Do not back-date to the first of the k required events.
* Forward-fill DC context to later bars; never back-fill.

# Debounce logic

* “k consecutive events” means you flip on event k’s bar.
* No partial anticipations. No marking earlier bars.

# Preprocessing

* Scaling, standardization, and thresholds must be estimated on past data only (per refit window).
* No centered filters. Use EMA/SMA with past-only windows.

# Joins and usage

* Join `reg_*` onto OHLCV with `ffill`. Never `bfill`.
* Use `.shift(1)` for entries, exits, and sizing logic.
* If you trade at bar close, document that and test against close-based fills only.

# Walk-forward discipline

* Split time: train → validate → test.
* For each refit date, freeze params, then generate `reg_*` forward until next refit.
* Log `model_version` and `fit_end_time` with each row to audit.

# Edge cases to test

* Bars with intra-bar DC flips. Decide a convention: event timestamp = bar close.
* Missing ticks or session gaps. Ensure the filter carries state without peeking.
* Parameter changes at refits. Verify no retroactive changes to prior rows.

# Quick self-checks

* Recompute the indicator in a single left-to-right pass; hashes of past rows must never change.
* Replace future returns with noise; the indicator for a given date must be identical.
* Compare online vs batch code that processes only up-to-t data; results match to the bar.

If you follow the above, the indicator is free of lookahead bias and future leak. The only unsafe variants are back-dating confirmations, using smoothed/Viterbi states, centered transforms, or full-sample normalization.

