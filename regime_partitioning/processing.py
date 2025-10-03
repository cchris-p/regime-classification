import pandas as pd
import numpy as np

import ruptures as rpt
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler


# -------------------------
# 1) PELT on rate_diff_2y and rv_20d (separately)
# -------------------------
def pelt_changepoints(series: pd.Series, penalty: float, min_size: int = 20):
    """Return changepoint indices (end of segments) using PELT with L2 cost."""
    x = series.dropna().values.astype(float)
    algo = rpt.Pelt(model="l2", min_size=min_size).fit(x)
    # returns segment endpoints; last endpoint = len(x)
    bkpts = algo.predict(pen=penalty)
    # convert to index positions aligned to original series
    idx = series.dropna().index
    cps = [idx[i - 1] for i in bkpts[:-1]]  # exclude final endpoint
    return cps


# -------------------------
# 2) 2-state Gaussian HMM on [ret, rv_20d]
# -------------------------
def fit_2state_hmm(
    df: pd.DataFrame,
    cols=("ret", "rv_20d"),
    n_states=2,
    n_init=10,
    max_iter=200,
    random_state=0,
):
    """Fit 2-state diagonal-cov GaussianHMM on standardized features."""
    X = df.loc[:, cols].dropna().astype(float).values
    scaler = StandardScaler().fit(X)
    Xz = scaler.transform(X)

    # hmmlearn uses full cov by default; set covariance_type='diag' for diagonal
    hmm = GaussianHMM(
        n_components=n_states,
        covariance_type="diag",
        n_iter=max_iter,
        tol=1e-4,
        random_state=random_state,
        init_params="stmc",  # learn startprob, transmat, means, covars
        params="stmc",
        verbose=False,
    )

    # Multiple random restarts for robustness
    best_model, best_score = None, -np.inf
    for seed in range(n_init):
        hmm.random_state = seed
        hmm.fit(Xz)
        score = hmm.score(Xz)
        if score > best_score:
            best_model, best_score = hmm, score

    # Decode
    z = best_model.predict(Xz)  # Viterbi path (0/1)
    post = best_model.predict_proba(Xz)  # state posteriors gamma_tk
    # Map results back to index
    out = df.loc[:, cols].dropna().copy()
    out["state"] = z
    out["p_state0"] = post[:, 0]
    out["p_state1"] = post[:, 1]

    # Optional labeling: risk-on = lower rv, higher ret
    mu = best_model.means_
    # means are in standardized space; still fine for ordering
    risk_on = int((mu[0, 1] < mu[1, 1]) and (mu[0, 0] > mu[1, 0]))
    risk_off = 1 - risk_on
    out["regime"] = np.where(out["state"] == risk_on, "risk_on", "risk_off")

    return best_model, scaler, out
