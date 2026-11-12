#!/usr/bin/env python3
"""Emit the daily online regime signal for one symbol.

Contract (read-only, consumable by ``TJB-073``)::

    {
      "as_of": "YYYY-MM-DD",
      "symbol": "NZDUSD",
      "regime": "risk_on" | "risk_off" | "NO DATA",
      "p_states": {"risk_on": float|null, "risk_off": float|null},
      "classifier_version": "regime-partitioning-online-v1",
      "fast_flag": bool,
      "fast_reasons": [str, ...],
      "stale": bool,
      "no_data": bool
    }

Run from the repo root:

    python scripts/emit_regime_signal.py --symbol NZDUSD
    python scripts/emit_regime_signal.py --symbol NZDUSD --as-of 2024-12-30 --out /tmp/sig.json
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from regime_partitioning.online import RegimeStreamingDetector, WalkForwardHMMTracker  # noqa: E402

EXPORT_ROOT = REPO / "exports" / "forex"
MODEL_ROOT = EXPORT_ROOT / "models"
CLASSIFIER_VERSION = "regime-partitioning-online-v1"


def _load_bars(symbol: str, as_of: pd.Timestamp, source: str) -> pd.DataFrame:
    if source == "export":
        path = EXPORT_ROOT / f"{symbol.upper()}_regime_ohlcv.csv"
        if not path.exists():
            raise FileNotFoundError(f"missing regime export: {path}")
        df = pd.read_csv(path, parse_dates=["datetime"]).set_index("datetime")
        df = df[df.index <= as_of]
        return df[["open", "high", "low", "close", "volume"]]
    if source == "dukascopy":
        sys.path.insert(0, "/home/matrillo/apps/jupyter-notebooks/trading_utils/src")
        from trading_utils.get_forex_data import get_forex_data_by_pair

        os.environ.setdefault(
            "DUKASCOPY_ROOT", "/home/matrillo/apps/jupyter-notebooks/histdata/dukascopy"
        )
        start = (as_of - pd.Timedelta(days=800)).strftime("%Y-%m-%d")
        df = get_forex_data_by_pair(
            symbol=symbol,
            start_date=start,
            end_date=as_of.strftime("%Y-%m-%d"),
            granularity="D",
            source="dukascopy",
        )
        return df[df.index <= as_of]
    raise ValueError(f"unknown source: {source}")


def build_signal(
    symbol: str,
    as_of: pd.Timestamp,
    source: str = "export",
) -> dict:
    bars = _load_bars(symbol, as_of, source)
    if bars.empty:
        return {
            "as_of": as_of.strftime("%Y-%m-%d"),
            "symbol": symbol.upper(),
            "regime": "NO DATA",
            "p_states": {"risk_on": None, "risk_off": None},
            "classifier_version": CLASSIFIER_VERSION,
            "fast_flag": False,
            "fast_reasons": [],
            "stale": True,
            "no_data": True,
        }
    tracker = WalkForwardHMMTracker(
        min_train=252,
        retrain_interval=20,
        n_init=10,
        random_state=0,
        classifier_version=CLASSIFIER_VERSION,
    )
    detector = RegimeStreamingDetector(
        tracker=tracker,
        bar_minutes=1440,
        classifier_version=CLASSIFIER_VERSION,
    )
    signal = None
    for ts, row in bars.iterrows():
        bar = {
            "t": ts,
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row.get("volume") or 0.0),
        }
        signal = detector.on_bar(bar, as_of=ts)
    assert signal is not None
    last_bar = pd.Timestamp(bars.index[-1]).normalize()
    stale = (pd.Timestamp(as_of).normalize() - last_bar).days > 2
    out = {
        "as_of": pd.Timestamp(as_of).strftime("%Y-%m-%d"),
        "symbol": symbol.upper(),
        "regime": signal["regime"],
        "p_states": signal["p_states"],
        "classifier_version": signal["classifier_version"],
        "fast_flag": signal["fast_flag"],
        "fast_reasons": signal["fast_reasons"],
        "stale": bool(stale),
        "no_data": bool(signal["no_data"]),
        "last_bar": last_bar.strftime("%Y-%m-%d"),
        "cadence": "D1",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--as-of", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    ap.add_argument("--source", choices=["export", "dukascopy"], default="export")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    signal = build_signal(
        args.symbol,
        pd.Timestamp(args.as_of),
        source=args.source,
    )
    out = Path(args.out) if args.out else EXPORT_ROOT / "signals" / f"{args.symbol.upper()}_regime_signal.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(signal, fh, indent=2)
        fh.write("\n")
    print(json.dumps(signal, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
