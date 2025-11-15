import os
import json
import argparse
import numpy as np
import pandas as pd
from regime_partitioning.processing import fit_2state_hmm
from regime_partitioning.online.dc import DCUpdater
from get_forex_data import get_forex_data_by_pair
import joblib


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="EURUSD")
    p.add_argument("--granularity", default="D")
    p.add_argument("--start", default="2015-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--out_base", default="artifacts/eurusd")
    p.add_argument("--dc_theta_pct", type=float, default=0.4)
    args = p.parse_args()

    df = get_forex_data_by_pair(
        symbol=args.symbol,
        granularity=args.granularity,
        start_date=args.start,
        end_date=args.end,
    )
    df = df.sort_index()
    close = df["close"].astype(float)
    ret = np.log(close / close.shift(1))
    rv_20d = ret.rolling(20).std() * np.sqrt(252.0)
    feat = pd.DataFrame({"ret": ret, "rv_20d": rv_20d}).dropna()

    model, scaler, hmm_df = fit_2state_hmm(feat, cols=("ret", "rv_20d"))

    os.makedirs(os.path.dirname(args.out_base), exist_ok=True)
    joblib.dump((model, scaler), f"{args.out_base}_hmm.pkl")

    p1 = hmm_df["p_state1"].astype(float)
    p1 = p1.reindex(close.index).ffill()

    dc = DCUpdater(theta_pct=float(args.dc_theta_pct))
    tmv0 = []
    t0 = []
    tmv1 = []
    t1 = []
    for t, price in close.items():
        evs = dc.update(pd.Timestamp(t), float(price))
        if len(evs) == 0:
            continue
        pv = p1.loc[t]
        if not np.isfinite(pv):
            continue
        y = 1 if pv >= 0.5 else 0
        for ev in evs:
            if y == 1:
                tmv1.append(float(ev.tmv))
                t1.append(int(ev.tlen))
            else:
                tmv0.append(float(ev.tmv))
                t0.append(int(ev.tlen))

    n0 = len(t0)
    n1 = len(t1)
    nt = n0 + n1
    if nt == 0:
        priors = {0: 0.5, 1: 0.5}
        cond = {
            0: {"tmv": (0.0, 1.0), "tlen": (0.0, 1.0)},
            1: {"tmv": (0.0, 1.0), "tlen": (0.0, 1.0)},
        }
    else:
        p0 = max(n0 / nt, 1e-6)
        p1v = max(n1 / nt, 1e-6)
        priors = {0: float(p0), 1: float(p1v)}

        def ms(x):
            x = np.asarray(x, dtype=float)
            if x.size == 0:
                return 0.0, 1.0
            m = float(np.mean(x))
            s = float(np.std(x))
            if not np.isfinite(s) or s <= 0:
                s = 1.0
            return m, s

        cond = {
            0: {"tmv": ms(tmv0), "tlen": ms(t0)},
            1: {"tmv": ms(tmv1), "tlen": ms(t1)},
        }

    with open(f"{args.out_base}_nb.json", "w") as f:
        json.dump({"priors": priors, "cond_params": cond}, f)


if __name__ == "__main__":
    main()
