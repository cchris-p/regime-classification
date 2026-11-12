#!/usr/bin/env python3
"""Backfill online labels and reconcile them against the offline ``vol_state``.

The online tracker is causal: the HMM is fitted on the first ``min_train`` bars
and then forward-filtered. The comparison is therefore out-of-sample on the
reconciliation window, which is the honest way to report agreement with the
walk-forward offline labels.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from regime_partitioning.online import WalkForwardHMMTracker  # noqa: E402

EXPORT_ROOT = REPO / "exports" / "forex"
OUT_ROOT = EXPORT_ROOT / "backfill"
CLASSIFIER_VERSION = "regime-partitioning-online-v1"
MIN_TRAIN = 252
TOLERANCE = 0.80


def backfill_symbol(symbol: str, min_train: int = MIN_TRAIN) -> dict:
    path = EXPORT_ROOT / f"{symbol.upper()}_regime_ohlcv.csv"
    df = pd.read_csv(path, parse_dates=["datetime"]).set_index("datetime")
    data = df[["ret", "rv_20d"]].dropna()
    if len(data) < min_train + 30:
        return {
            "symbol": symbol.upper(),
            "status": "no_data",
            "usable_rows": int(len(data)),
        }
    # Feed the export's own ret/rv_20d observations so the comparison isolates
    # the HMM refit schedule and label mapping, not the feature pipeline. The
    # live emitter computes ret/rv_20d from bars (see streaming.py).
    tracker = WalkForwardHMMTracker(
        min_train=min_train, retrain_interval=20, n_init=10, random_state=0,
        classifier_version=CLASSIFIER_VERSION,
    )
    rows = []
    for ts, row in df.iterrows():
        obs = row[["ret", "rv_20d"]]
        if obs.isna().any():
            scored = {"regime": "NO DATA", "no_data": True}
        else:
            scored = tracker.score_step(obs)
        rows.append(
            {
                "datetime": ts,
                "online_regime": scored["regime"],
                "online_no_data": scored["no_data"],
                "offline_vol_state": row.get("vol_state"),
                "offline_final_regime": row.get("final_regime"),
            }
        )
    out = pd.DataFrame(rows).set_index("datetime")
    eval_mask = (
        (out.index >= data.index[min_train])
        & (out["online_regime"].isin(["risk_on", "risk_off"]))
        & (out["offline_vol_state"].isin(["risk_on", "risk_off"]))
    )
    ev = out.loc[eval_mask]
    n = int(len(ev))
    if n == 0:
        return {"symbol": symbol.upper(), "status": "no_overlap", "usable_rows": len(data)}
    agree = float((ev["online_regime"] == ev["offline_vol_state"]).mean())
    confusion = pd.crosstab(ev["online_regime"], ev["offline_vol_state"]).to_dict()
    result = {
        "symbol": symbol.upper(),
        "status": "ok",
        "classifier_version": CLASSIFIER_VERSION,
        "min_train": int(min_train),
        "usable_rows": int(len(data)),
        "overlap_rows": n,
        "agreement": round(agree, 4),
        "tolerance": TOLERANCE,
        "within_tolerance": bool(agree >= TOLERANCE),
        "confusion": {str(k): {str(kk): int(vv) for kk, vv in v.items()} for k, v in confusion.items()},
        "evaluation_window": [
            ev.index.min().strftime("%Y-%m-%d"),
            ev.index.max().strftime("%Y-%m-%d"),
        ],
    }
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    ev.to_csv(OUT_ROOT / f"{symbol.upper()}_backfill.csv")
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbols", nargs="*", default=["NZDUSD", "EURCAD", "EURCHF"])
    ap.add_argument("--min-train", type=int, default=MIN_TRAIN)
    args = ap.parse_args()
    results = [backfill_symbol(s, args.min_train) for s in args.symbols]
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    with open(OUT_ROOT / "reconciliation.json", "w") as fh:
        json.dump(results, fh, indent=2)
        fh.write("\n")
    print(json.dumps(results, indent=2))
    ok = all(r.get("status") != "ok" or r.get("within_tolerance") for r in results)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
