import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

TRADING_UTILS_ROOT = Path("/home/matrillo/apps/jupyter-notebooks")
if str(TRADING_UTILS_ROOT) not in sys.path:
    sys.path.insert(0, str(TRADING_UTILS_ROOT))

from regime_partitioning.datasets import fx_datasets
from regime_partitioning.processing import pelt_changepoints
from trading_utils.get_forex_data import get_forex_data_by_pair
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler


def compute_segment_ids(index, changepoints):
    cps = sorted(changepoints)
    segment_ids = []
    current_segment = 0
    cp_idx = 0
    n_cps = len(cps)
    current_cp = cps[cp_idx] if cp_idx < n_cps else None
    for ts in index:
        while current_cp is not None and ts > current_cp:
            current_segment += 1
            cp_idx += 1
            current_cp = cps[cp_idx] if cp_idx < n_cps else None
        segment_ids.append(current_segment)
    return pd.Series(segment_ids, index=index)


def label_macro_state(series, penalty, min_size=20, n_bins=3):
    s = series.dropna()
    if len(s) == 0:
        return pd.Series(index=series.index, dtype="float64")
    cps = pelt_changepoints(s, penalty=penalty, min_size=min_size)
    seg_ids = compute_segment_ids(s.index, cps)
    seg_means = s.groupby(seg_ids).mean()
    probs = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    bins = np.quantile(seg_means.values, probs)
    states = np.digitize(seg_means.values, bins)
    mapping = {seg: state for seg, state in zip(seg_means.index, states)}
    macro_clean = seg_ids.map(mapping)
    macro = pd.Series(index=series.index, dtype="float64")
    macro.loc[s.index] = macro_clean.values
    return macro


def walkforward_hmm_2state(
    df,
    cols=("ret", "rv_20d"),
    n_states=2,
    n_init=10,
    max_iter=200,
    min_train_size=252,
    retrain_interval=20,
    random_state=0,
):
    data = df.loc[:, cols].dropna()
    if len(data) < min_train_size:
        return pd.DataFrame(
            index=data.index,
            columns=["state", "p_state0", "p_state1", "regime"],
        )
    X_full = data.values.astype(float)
    n = X_full.shape[0]
    state = np.full(n, np.nan)
    p0 = np.full(n, np.nan)
    p1 = np.full(n, np.nan)
    regime = np.array([""] * n, dtype=object)
    current_model = None
    current_scaler = None
    last_fit_at = None
    risk_on_state = None
    for t in range(n):
        if t + 1 < min_train_size:
            continue
        need_refit = (
            current_model is None
            or last_fit_at is None
            or (t - last_fit_at) >= retrain_interval
        )
        if need_refit:
            X_train = X_full[:t].astype(float)
            scaler = StandardScaler().fit(X_train)
            X_train_z = scaler.transform(X_train)
            best_model = None
            best_score = -np.inf
            for seed in range(n_init):
                hmm = GaussianHMM(
                    n_components=n_states,
                    covariance_type="diag",
                    n_iter=max_iter,
                    tol=1e-4,
                    random_state=random_state + seed,
                    init_params="stmc",
                    params="stmc",
                    verbose=False,
                )
                hmm.fit(X_train_z)
                score = hmm.score(X_train_z)
                if score > best_score:
                    best_model = hmm
                    best_score = score
            mu = best_model.means_
            risk_on = int((mu[0, 1] < mu[1, 1]) and (mu[0, 0] > mu[1, 0]))
            risk_on_state = risk_on
            current_model = best_model
            current_scaler = scaler
            last_fit_at = t
        x_t = X_full[t : t + 1]
        x_t_z = current_scaler.transform(x_t)
        post = current_model.predict_proba(x_t_z)[0]
        z_t = int(np.argmax(post))
        state[t] = z_t
        p0[t] = post[0]
        p1[t] = post[1]
        if risk_on_state is not None and z_t == risk_on_state:
            regime[t] = "risk_on"
        else:
            regime[t] = "risk_off"
    out = pd.DataFrame(
        {
            "state": state,
            "p_state0": p0,
            "p_state1": p1,
            "regime": regime,
        },
        index=data.index,
    )
    out.replace("", np.nan, inplace=True)
    return out


def build_regime_dataset_for_symbol(symbol, export_dir):
    ds = fx_datasets[symbol]
    df_fx = ds["df_fx"].copy()
    s_yield = df_fx["rate_diff_2y"].dropna()
    s_cpi = df_fx["cpi_diff_core"].dropna()
    pen_yield = 3.0 * np.log(len(s_yield)) if len(s_yield) > 0 else 0.0
    pen_cpi = 3.0 * np.log(len(s_cpi)) if len(s_cpi) > 0 else 0.0
    macro_yield_state = label_macro_state(df_fx["rate_diff_2y"], penalty=pen_yield)
    macro_cpi_state = label_macro_state(df_fx["cpi_diff_core"], penalty=pen_cpi)
    hmm_out = walkforward_hmm_2state(df_fx, cols=("ret", "rv_20d"))
    df_reg = df_fx.join(
        [
            macro_yield_state.rename("macro_yield_state"),
            macro_cpi_state.rename("macro_cpi_state"),
            hmm_out,
        ],
        how="left",
    )
    macro_yield_cat = df_reg["macro_yield_state"].fillna(-1).astype(int).astype(str)
    macro_cpi_cat = df_reg["macro_cpi_state"].fillna(-1).astype(int).astype(str)
    df_reg["macro_state"] = "y" + macro_yield_cat + "_c" + macro_cpi_cat
    df_reg["vol_state"] = df_reg["regime"].fillna("unknown")
    df_reg["final_regime"] = df_reg["macro_state"] + "|" + df_reg["vol_state"]
    start_date = df_reg.index.min().strftime("%Y-%m-%d")
    end_date = df_reg.index.max().strftime("%Y-%m-%d")
    df_px = get_forex_data_by_pair(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        granularity="D",
    )
    df_px = df_px.sort_index()
    df_full = df_px.join(df_reg, how="left")
    os.makedirs(export_dir, exist_ok=True)
    out_path = os.path.join(export_dir, f"{symbol}_regime_ohlcv.csv")
    df_full.to_csv(out_path, index_label="datetime")
    return out_path


def main():
    project_root = os.path.dirname(os.path.abspath(__file__))
    export_dir = os.path.join(project_root, "exports", "forex")
    for symbol in fx_datasets.keys():
        build_regime_dataset_for_symbol(symbol, export_dir)


if __name__ == "__main__":
    main()
