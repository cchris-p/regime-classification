import argparse
import json
import sys

import joblib
import numpy as np
import pandas as pd

from get_forex_data import get_forex_data_by_pair
from regime_partitioning.online import RegimeStreamingDetector, WindowRule


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="EURUSD")
    p.add_argument("--granularity", default="D")
    p.add_argument("--start", default="2020-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--dc_theta_pct", type=float, default=0.4)
    p.add_argument("--theta_open", type=float, default=0.80)
    p.add_argument("--theta_close", type=float, default=0.50)
    p.add_argument("--k", type=int, default=2)
    p.add_argument("--k_out", type=int, default=2)
    p.add_argument("--L_min", type=int, default=2)
    p.add_argument("--hmm_path", default="artifacts/eurusd_hmm.pkl")
    p.add_argument("--nb_path", default=None)
    p.add_argument("--use_bayes", action="store_true")
    args = p.parse_args()

    (hmm, scaler) = joblib.load(args.hmm_path)

    priors = None
    cond = None
    if args.use_bayes and args.nb_path is not None:
        with open(args.nb_path, "r") as f:
            blob = json.load(f)
            priors = blob.get("priors")
            cond = blob.get("cond_params")

    rule = WindowRule(
        open_p=float(args.theta_open),
        close_p=float(args.theta_close),
        confirm_open=int(args.k),
        confirm_close=int(args.k_out),
        min_trends=int(args.L_min),
    )

    det = RegimeStreamingDetector(
        hmm,
        scaler,
        dc_theta_pct=float(args.dc_theta_pct),
        rule=rule,
        use_bayes=bool(args.use_bayes),
        bayes_priors=priors,
        bayes_cond_params=cond,
    )

    df = get_forex_data_by_pair(
        symbol=args.symbol,
        granularity=args.granularity,
        start_date=args.start,
        end_date=args.end,
    )
    df = df.sort_index()

    for t, row in df.iterrows():
        bar = {
            "t": pd.Timestamp(t),
            "open": float(row.get("open", np.nan)),
            "high": float(row.get("high", np.nan)),
            "low": float(row.get("low", np.nan)),
            "close": float(row.get("close", np.nan)),
            "volume": float(row.get("volume", 0.0)),
        }
        changed = det.on_bar(bar)
        if changed:
            for w in changed:
                if w.end is None:
                    sys.stdout.write(f"OPEN,{w.start.isoformat()},regime_2\n")
                else:
                    sys.stdout.write(f"CLOSE,{w.end.isoformat()},regime_2\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
