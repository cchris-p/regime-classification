from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


class HMMTracker:
    def __init__(
        self, model, scaler, feature_cols: Tuple[str, ...] = ("ret", "rv_20d")
    ):
        self.model = model
        self.scaler = scaler
        self.cols = feature_cols
        self._X: List[List[float]] = []

    def score_step(self, obs_row: pd.Series) -> Dict[str, float]:
        x = [float(obs_row.get(c)) for c in self.cols]
        if any(not np.isfinite(v) for v in x):
            if len(self._X) == 0:
                return {"p_state0": np.nan, "p_state1": np.nan, "map_state": np.nan}
            post = self.model.predict_proba(np.asarray(self._X))
            p0, p1 = float(post[-1, 0]), float(post[-1, 1])
            return {
                "p_state0": p0,
                "p_state1": p1,
                "map_state": int(np.argmax([p0, p1])),
            }

        Xz = self.scaler.transform(np.asarray([x]))
        if len(self._X) == 0:
            self._X = Xz.tolist()
        else:
            self._X.extend(Xz.tolist())
        post = self.model.predict_proba(np.asarray(self._X))
        p0, p1 = float(post[-1, 0]), float(post[-1, 1])
        return {"p_state0": p0, "p_state1": p1, "map_state": int(np.argmax([p0, p1]))}
