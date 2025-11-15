from __future__ import annotations

from math import log, pi
from typing import Dict, Tuple

import numpy as np


class NaiveBayesTracker:
    def __init__(
        self,
        priors: Dict[int, float],
        cond_params: Dict[int, Dict[str, Tuple[float, float]]],
    ):
        self.priors = {int(k): float(v) for k, v in priors.items()}
        self.params = cond_params

    def _gauss_ll(self, x: float, mu: float, sigma: float) -> float:
        if sigma <= 0 or not np.isfinite(sigma):
            return -np.inf
        z = (x - mu) / sigma
        return -0.5 * (z * z) - log(sigma) - 0.5 * log(2.0 * pi)

    def score_step(self, tmv: float, tlen: float) -> Dict[str, float]:
        lls = {}
        for cls, prior in self.priors.items():
            p = self.params.get(cls, {})
            mu_tmv, sd_tmv = p.get("tmv", (0.0, 1.0))
            mu_t, sd_t = p.get("tlen", (0.0, 1.0))
            ll = self._gauss_ll(float(tmv), float(mu_tmv), float(sd_tmv))
            ll += self._gauss_ll(float(tlen), float(mu_t), float(sd_t))
            ll += log(max(prior, 1e-12))
            lls[int(cls)] = ll
        m = max(lls.values())
        exps = {k: np.exp(v - m) for k, v in lls.items()}
        Z = sum(exps.values())
        p0 = exps.get(0, 0.0) / Z if Z > 0 else np.nan
        p1 = exps.get(1, 0.0) / Z if Z > 0 else np.nan
        return {"p_regime1": p0, "p_regime2": p1}
