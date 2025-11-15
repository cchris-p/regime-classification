from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from .dc import DCUpdater
from .features import FeatureBuilder
from .hmm_tracker import HMMTracker
from .windows import WindowRule, WindowStateMachine

try:
    from .bayes_tracker import NaiveBayesTracker
except Exception:
    NaiveBayesTracker = None


class RegimeStreamingDetector:
    def __init__(
        self,
        hmm_model,
        scaler,
        dc_theta_pct: float = 0.4,
        rule: WindowRule = WindowRule(),
        use_bayes: bool = False,
        bayes_priors=None,
        bayes_cond_params=None,
    ):
        self.features = FeatureBuilder()
        self.dc = DCUpdater(theta_pct=dc_theta_pct)
        self.tracker = HMMTracker(hmm_model, scaler)
        self.windows = WindowStateMachine(rule)
        self.use_bayes = use_bayes
        self.nb = (
            NaiveBayesTracker(bayes_priors, bayes_cond_params)
            if use_bayes
            and NaiveBayesTracker is not None
            and bayes_priors is not None
            and bayes_cond_params is not None
            else None
        )
        self._last_p1 = np.nan

    def on_bar(self, bar: dict) -> List[Window]:
        t = pd.Timestamp(bar.get("t"))
        price = float(bar.get("close")) if bar.get("close") is not None else np.nan
        dc_events = self.dc.update(t, price) if np.isfinite(price) else []
        p1 = np.nan
        if self.nb is not None:
            if len(dc_events) > 0:
                ev = dc_events[-1]
                s = self.nb.score_step(ev.tmv, ev.tlen)
                p1 = float(s.get("p_regime2")) if s is not None else np.nan
                self._last_p1 = p1
            else:
                p1 = self._last_p1
        else:
            feat = self.features.on_bar(bar)
            score = self.tracker.score_step(feat)
            p1 = score.get("p_state1")
        changed: List[Window] = []
        if len(dc_events) > 0 and np.isfinite(p1):
            for _ in dc_events:
                changed.extend(self.windows.on_prob(t, float(p1), True))
        return changed


def build_regime_indicator(
    close: pd.Series,
    hmm_model,
    scaler,
    theta_open: float = 0.80,
    theta_close: float = 0.50,
    k: int = 2,
    k_out: int = 2,
    L_min: int = 2,
    dc_theta_pct: float = 0.4,
) -> pd.DataFrame:
    rule = WindowRule(
        open_p=float(theta_open),
        close_p=float(theta_close),
        confirm_open=int(k),
        confirm_close=int(k_out),
        min_trends=int(L_min),
    )
    dc = DCUpdater(theta_pct=dc_theta_pct)
    fb = FeatureBuilder()
    tracker = HMMTracker(hmm_model, scaler)
    sm = WindowStateMachine(rule)

    rows = []
    window_id: Optional[int] = None
    age = 0
    last_dc_tmv = np.nan
    last_dc_T = np.nan
    last_dc_R = np.nan

    for t, price in close.items():
        t = pd.Timestamp(t)
        bar = {"t": t, "close": float(price)}
        dc_events = dc.update(t, float(price)) if np.isfinite(price) else []
        feat = fb.on_bar(bar)
        score = tracker.score_step(feat)
        p0 = score.get("p_state0")
        p1 = score.get("p_state1")
        map_state = np.nan
        if np.isfinite(p0) and np.isfinite(p1):
            map_state = int(np.argmax([p0, p1]))
        changed = []
        if len(dc_events) > 0 and np.isfinite(p1):
            for ev in dc_events:
                last_dc_tmv = float(ev.tmv)
                last_dc_T = int(ev.tlen)
                last_dc_R = float(ev.r)
                changed.extend(sm.on_prob(t, float(p1), True))
        reg_open = 0
        reg_close = 0
        if len(changed) > 0:
            for w in changed:
                if w.end is None:
                    reg_open = 1
                    window_id = 1 if window_id is None else window_id + 1
                    age = 0
                else:
                    reg_close = 1
                    window_id = None
                    age = 0
        increment_age = (window_id is not None) and (reg_open == 0)
        rows.append(
            {
                "t": t,
                "reg_state": map_state,
                "reg_p0": p0,
                "reg_p1": p1,
                "reg_open": reg_open,
                "reg_close": reg_close,
                "reg_window_id": window_id,
                "reg_age": age if window_id is not None else 0,
                "reg_conf": (
                    np.nanmax([p0, p1])
                    if np.isfinite(p0) and np.isfinite(p1)
                    else np.nan
                ),
                "dc_tmv": last_dc_tmv,
                "dc_T": last_dc_T,
                "dc_R": last_dc_R,
                "dc_event_bar": int(len(dc_events) > 0),
            }
        )
        if increment_age:
            age += 1

    out = pd.DataFrame.from_records(rows).set_index("t")
    return out
