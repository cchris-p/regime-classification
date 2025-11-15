import argparse
import itertools
import numpy as np
import pandas as pd
import joblib

from get_forex_data import get_forex_data_by_pair
from regime_partitioning.online import build_regime_indicator


def contiguous_segments(active):
    runs = []
    start = None
    length = 0
    for i, v in enumerate(active):
        if v and start is None:
            start = i
            length = 1
        elif v and start is not None:
            length += 1
        elif (not v) and start is not None:
            runs.append((start, length))
            start = None
            length = 0
    if start is not None:
        runs.append((start, length))
    return runs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="EURUSD")
    p.add_argument("--granularity", default="D")
    p.add_argument("--start", default="2018-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--hmm_path", default="artifacts/eurusd_hmm.pkl")
    p.add_argument("--dc_list", default="0.3,0.4,0.5")
    p.add_argument("--open_list", default="0.75,0.80,0.85")
    p.add_argument("--close_list", default="0.45,0.50,0.55,0.60")
    p.add_argument("--k_list", default="1,2,3")
    p.add_argument("--kout_list", default="1,2,3")
    p.add_argument("--Lmin_list", default="1,2,3")
    args = p.parse_args()

    (hmm, scaler) = joblib.load(args.hmm_path)

    df = get_forex_data_by_pair(
        symbol=args.symbol,
        granularity=args.granularity,
        start_date=args.start,
        end_date=args.end,
    ).sort_index()
    close = df["close"].astype(float)

    dc_vals = [float(x) for x in args.dc_list.split(",") if x]
    open_vals = [float(x) for x in args.open_list.split(",") if x]
    close_vals = [float(x) for x in args.close_list.split(",") if x]
    k_vals = [int(x) for x in args.k_list.split(",") if x]
    kout_vals = [int(x) for x in args.kout_list.split(",") if x]
    Lmin_vals = [int(x) for x in args.Lmin_list.split(",") if x]

    print("dc_theta_pct,open_p,close_p,k,k_out,L_min,windows,mean_len,median_len")
    for dc_theta_pct, open_p, close_p, k, k_out, L_min in itertools.product(
        dc_vals, open_vals, close_vals, k_vals, kout_vals, Lmin_vals
    ):
        reg = build_regime_indicator(
            close,
            hmm,
            scaler,
            theta_open=open_p,
            theta_close=close_p,
            k=k,
            k_out=k_out,
            L_min=L_min,
            dc_theta_pct=dc_theta_pct,
        )
        active = reg["reg_window_id"].notna().values.tolist()
        runs = contiguous_segments(active)
        lengths = [l for _, l in runs]
        windows = int(reg["reg_open"].sum())
        mean_len = float(np.mean(lengths)) if lengths else 0.0
        med_len = float(np.median(lengths)) if lengths else 0.0
        print(
            f"{dc_theta_pct},{open_p},{close_p},{k},{k_out},{L_min},{windows},{mean_len:.1f},{med_len:.1f}"
        )


if __name__ == "__main__":
    main()
