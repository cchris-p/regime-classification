#!/usr/bin/env python3
"""Fit and persist the online HMM model/scaler artifacts.

The offline walk-forward fitter retrains every 20 bars. The online emitter needs
one persisted model per symbol; this script fits the 2-state diagonal Gaussian
HMM on the symbol's regime export and writes ``<SYMBOL>_hmm.pkl``.

Run from the repo root:

    python scripts/fit_regime_models.py --symbols NZDUSD EURCAD EURCHF
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from regime_partitioning.processing import fit_2state_hmm  # noqa: E402

EXPORT_ROOT = REPO / "exports" / "forex"
MODEL_ROOT = EXPORT_ROOT / "models"


def fit_symbol(symbol: str, export_root: Path, model_root: Path) -> Path:
    path = export_root / f"{symbol.upper()}_regime_ohlcv.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing regime export: {path}")
    df = pd.read_csv(path, parse_dates=["datetime"]).set_index("datetime")
    data = df[["ret", "rv_20d"]].dropna()
    if len(data) < 252:
        raise ValueError(f"{symbol}: only {len(data)} usable rows, need >= 252")
    model, scaler, _ = fit_2state_hmm(data)
    model_root.mkdir(parents=True, exist_ok=True)
    out = model_root / f"{symbol.upper()}_hmm.pkl"
    with open(out, "wb") as fh:
        pickle.dump({"model": model, "scaler": scaler, "trained_through": str(data.index.max())}, fh)
    print(f"{symbol}: wrote {out} ({len(data)} rows through {data.index.max().date()})")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbols", nargs="*", default=["NZDUSD", "EURCAD", "EURCHF"])
    ap.add_argument("--export-root", default=str(EXPORT_ROOT))
    ap.add_argument("--model-root", default=str(MODEL_ROOT))
    args = ap.parse_args()
    for symbol in args.symbols:
        fit_symbol(symbol, Path(args.export_root), Path(args.model_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
