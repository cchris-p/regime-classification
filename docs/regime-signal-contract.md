# Regime Signal Contract

**Card:** `REGIME-002` (producer), `TJB-073` (consumer)
**Classifier version:** `regime-partitioning-online-v1`
**Taxonomy:** `regime-taxonomy-v1`
**Status:** stable contract — 2026-11-12

The online regime layer emits a **read-only** JSON signal. It never places,
sizes, or flattens an order. Consumers must treat a missing, stale, or
`NO DATA` signal as "do not act on the regime" and fall back to the independent
tail-risk controls (`docs/ltr-tail-risk-policy.md`).

## Signal payload

```json
{
  "as_of": "2024-12-30",
  "symbol": "NZDUSD",
  "regime": "risk_off",
  "p_states": {"risk_on": 0.08, "risk_off": 0.92},
  "classifier_version": "regime-partitioning-online-v1",
  "fast_flag": false,
  "fast_reasons": [],
  "stale": false,
  "no_data": false,
  "last_bar": "2024-12-30",
  "cadence": "D1",
  "generated_at": "2026-11-12T22:20:00Z"
}
```

| Field | Type | Meaning |
|---|---|---|
| `as_of` | `YYYY-MM-DD` | Evaluation date. The signal is only valid for this date. |
| `symbol` | string | Instrument. |
| `regime` | `risk_on` \| `risk_off` \| `NO DATA` | Slow HMM label (`vol_state`) at `as_of`. |
| `p_states` | object | Raw state posteriors. `risk_on` + `risk_off` sum to ~1; both `null` when `no_data`. |
| `classifier_version` | string | Version of the feature/HMM/label logic. A change is a new version. |
| `fast_flag` | bool | Fast-path "conditions just changed" flag. Independent of the slow label. |
| `fast_reasons` | array[string] | Which fast triggers fired (rv spike, ADR usage, gap, spread). |
| `stale` | bool | True when the last bar is older than 2× the daily cadence (or the source is unavailable). |
| `no_data` | bool | True when there is no usable observation (warmup/missing features). |
| `last_bar` | `YYYY-MM-DD` | Last bar actually consumed. |
| `cadence` | string | `D1` for the current production cadence. |
| `generated_at` | ISO-8601 UTC | Emission time (provenance only; not the decision time). |

## Semantics

- **Two-speed design.** The slow label is the HMM `vol_state`; the fast flag is
  an intraday conditions-changed detector. A seeded vol spike raises
  `fast_flag` while the slow label remains stable (tested).
- **Hysteresis.** The label does not flip on a single bar: it requires
  `confirm_open`/`confirm_close` consecutive bars, the probability threshold,
  and a minimum dwell (`regime_partitioning/online/windows.py`).
- **Deterministic and versioned.** The HMM refit schedule mirrors the offline
  walk-forward fitter (`min_train=252`, `retrain_interval=20`, `n_init=10`,
  `random_state=0`). A change to any of these is a new `classifier_version`.
- **Staleness.** A consumer must reject `stale=true` or a `last_bar` older than
  the declared cadence. A stale signal is never silently reused.
- **`NO DATA` is explicit.** During warmup or when features are missing, the
  signal says so; it does not guess a label.

## Fast-path thresholds

Any trigger, after confirmation of **2 bars or 1 hour (whichever is longer)**:

- `rv_20d > 2×` its trailing 126-session median;
- ADR usage `> 100%` of the trailing 20-session ADR;
- session gap `> 1×` ADR;
- spread `> 2×` the trailing 20-session median.

## Reconciliation with the offline export

`scripts/backfill_regime_labels.py` replays the export and compares the online
`risk_on`/`risk_off` label with the offline `vol_state` on the post-warmup
overlap. `vol_state` is the HMM component of the offline `final_regime`
(`final_regime = macro_state|vol_state`); the online signal in this version
carries the HMM component only, so the reconciliation is against that component
and the macro component remains an offline evidence layer. The script writes
`exports/forex/backfill/reconciliation.json` with the agreement, confusion
matrix, tolerance (`0.80`), and evaluation window. The backfill feeds the
export's own `ret`/`rv_20d` observations to isolate the HMM refit schedule from
the live feature pipeline; the live emitter computes `ret`/`rv_20d` from bars.

The reconciliation is **causal** (the tracker only sees the past) and the
comparison window starts after the first `252` valid observations. Result
(2026-11-12): `1.000` agreement on NZDUSD (725 rows), EURCAD (961), EURCHF
(1028), all within the `0.80` tolerance.

## Canonical JSON schema

`docs/regime-signal.schema.json` (JSON Schema draft 2020-12) is the
machine-readable contract. Consumers may reject a payload that fails it.

## Producer and consumer

- Producer: `scripts/emit_regime_signal.py` (this repo).
- Consumer: `trade-journal` `TJB-073` (lane health) and the live check-in loop.
- Persistence: `exports/forex/signals/<SYMBOL>_regime_signal.json`.
- The signal is **advisory**. A pause/reduce/flatten action still requires the
  operator approval path in `docs/regime-recalibration-policy.md`.

## Regenerate / run

```bash
# fit optional fast-start artifacts (the emitter builds its own causal model)
python scripts/fit_regime_models.py --symbols NZDUSD EURCAD EURCHF
# emit (historical replay or current)
python scripts/emit_regime_signal.py --symbol NZDUSD --as-of 2024-12-30
# reconcile online vs offline
python scripts/backfill_regime_labels.py --symbols NZDUSD EURCAD EURCHF
```
