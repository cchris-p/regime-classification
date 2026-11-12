"""Online HMM forward-filter tracker.

Wraps a fitted two-state diagonal Gaussian HMM plus its scaler. The tracker is
causal: each ``score_step`` standardizes only the observation history seen so
far and returns the posterior of the last observation, which is the filtered
probability at the current bar (not the Viterbi path).
"""

from __future__ import annotations

import pickle
from collections import deque
from pathlib import Path
from typing import Deque, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def _risk_on_state(model) -> int:
    """Map the HMM state index with lower vol / higher mean return to risk-on."""
    mu = model.means_
    return int((mu[0, 1] < mu[1, 1]) and (mu[0, 0] > mu[1, 0]))


class HMMTracker:
    def __init__(
        self,
        model,
        scaler,
        feature_cols: Sequence[str] = ("ret", "rv_20d"),
        history_cap: int = 2000,
        classifier_version: str = "regime-partitioning-online-v1",
    ) -> None:
        self.model = model
        self.scaler = scaler
        self.cols = tuple(feature_cols)
        self.history_cap = int(history_cap)
        self.classifier_version = classifier_version
        self._history: Deque[np.ndarray] = deque(maxlen=self.history_cap)
        self.risk_on_state = _risk_on_state(model)

    @classmethod
    def from_artifact(cls, path: str | Path, **kwargs) -> "HMMTracker":
        with open(path, "rb") as fh:
            payload = pickle.load(fh)
        if isinstance(payload, dict):
            return cls(payload["model"], payload["scaler"], **kwargs)
        model, scaler = payload
        return cls(model, scaler, **kwargs)

    def score_step(self, obs_row: pd.Series) -> Dict[str, object]:
        vals = []
        for c in self.cols:
            v = obs_row.get(c)
            if v is None or pd.isna(v):
                return {
                    "p_state0": None,
                    "p_state1": None,
                    "map_state": None,
                    "regime": "NO DATA",
                    "no_data": True,
                }
            vals.append(float(v))
        self._history.append(np.asarray(vals, dtype=float))
        X = np.asarray(self._history, dtype=float)
        Xz = self.scaler.transform(X)
        post = self.model.predict_proba(Xz)[-1]
        state = int(np.argmax(post))
        return {
            "p_state0": float(post[0]),
            "p_state1": float(post[1]),
            "map_state": state,
            "regime": "risk_on" if state == self.risk_on_state else "risk_off",
            "no_data": False,
        }

    def reset_history(self, observations: Optional[np.ndarray] = None) -> None:
        self._history.clear()
        if observations is not None:
            for row in observations:
                self._history.append(np.asarray(row, dtype=float))

    @property
    def history_len(self) -> int:
        return len(self._history)


class WalkForwardHMMTracker:
    """Causal tracker that mirrors the offline walk-forward HMM refit schedule.

    The offline export trains a 2-state HMM on ``X[:t]``, refits every
    ``retrain_interval`` observations, and scores observation ``t`` with a
    single-row posterior. This tracker reproduces that schedule online, so its
    labels are reconcilable with the export's ``vol_state`` by construction
    rather than by a different model.
    """

    def __init__(
        self,
        feature_cols: Sequence[str] = ("ret", "rv_20d"),
        min_train: int = 252,
        retrain_interval: int = 20,
        n_init: int = 10,
        max_iter: int = 200,
        random_state: int = 0,
        classifier_version: str = "regime-partitioning-online-v1",
    ) -> None:
        self.cols = tuple(feature_cols)
        self.min_train = int(min_train)
        self.retrain_interval = int(retrain_interval)
        self.n_init = int(n_init)
        self.max_iter = int(max_iter)
        self.random_state = int(random_state)
        self.classifier_version = classifier_version
        self._history: list[np.ndarray] = []
        self._model = None
        self._scaler = None
        self._last_fit_at: Optional[int] = None
        self.risk_on_state: Optional[int] = None

    def _fit(self, X_train: np.ndarray):
        from hmmlearn.hmm import GaussianHMM
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler().fit(X_train)
        Xz = scaler.transform(X_train)
        best_model = None
        best_score = -np.inf
        for seed in range(self.n_init):
            hmm = GaussianHMM(
                n_components=2,
                covariance_type="diag",
                n_iter=self.max_iter,
                tol=1e-4,
                random_state=self.random_state + seed,
                init_params="stmc",
                params="stmc",
                verbose=False,
            )
            hmm.fit(Xz)
            score = hmm.score(Xz)
            if score > best_score:
                best_model = hmm
                best_score = score
        return best_model, scaler

    def score_step(self, obs_row: pd.Series) -> Dict[str, object]:
        vals = []
        for c in self.cols:
            v = obs_row.get(c)
            if v is None or pd.isna(v):
                return {
                    "p_state0": None,
                    "p_state1": None,
                    "map_state": None,
                    "regime": "NO DATA",
                    "no_data": True,
                }
            vals.append(float(v))
        x = np.asarray(vals, dtype=float)
        t = len(self._history)
        if t + 1 < self.min_train:
            self._history.append(x)
            return {
                "p_state0": None,
                "p_state1": None,
                "map_state": None,
                "regime": "NO DATA",
                "no_data": True,
            }
        need_refit = (
            self._model is None
            or self._last_fit_at is None
            or (t - self._last_fit_at) >= self.retrain_interval
        )
        if need_refit:
            X_train = np.asarray(self._history, dtype=float)
            self._model, self._scaler = self._fit(X_train)
            mu = self._model.means_
            self.risk_on_state = int((mu[0, 1] < mu[1, 1]) and (mu[0, 0] > mu[1, 0]))
            self._last_fit_at = t
        self._history.append(x)
        xz = self._scaler.transform(x.reshape(1, -1))
        post = self._model.predict_proba(xz)[0]
        state = int(np.argmax(post))
        return {
            "p_state0": float(post[0]),
            "p_state1": float(post[1]),
            "map_state": state,
            "regime": "risk_on" if state == self.risk_on_state else "risk_off",
            "no_data": False,
        }

    @property
    def history_len(self) -> int:
        return len(self._history)
