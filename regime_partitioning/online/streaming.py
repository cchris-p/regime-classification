"""Single ``on_bar`` facade for the two-speed online regime detector."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

from .dc import DCUpdater
from .fast_path import FastPathDetector
from .features import FeatureBuilder
from .hmm_tracker import HMMTracker
from .windows import Window, WindowRule, WindowStateMachine


@dataclass
class RegimeSignal:
    as_of: pd.Timestamp
    regime: str
    p_states: Dict[str, Optional[float]]
    classifier_version: str
    fast_flag: bool
    fast_reasons: List[str]
    stale: bool = False
    no_data: bool = False


class RegimeStreamingDetector:
    def __init__(
        self,
        hmm_model=None,
        scaler=None,
        tracker: Optional[HMMTracker] = None,
        dc_theta_pct: float = 0.4,
        rule: Optional[WindowRule] = None,
        bar_minutes: Optional[float] = None,
        classifier_version: str = "regime-partitioning-online-v1",
    ) -> None:
        self.features = FeatureBuilder()
        self.dc = DCUpdater(theta_pct=dc_theta_pct)
        self.tracker = tracker or HMMTracker(
            hmm_model, scaler, classifier_version=classifier_version
        )
        self.windows = WindowStateMachine(rule or WindowRule())
        self.fast_path = FastPathDetector(bar_minutes=bar_minutes)
        self.classifier_version = classifier_version
        self._last_windows: List[Window] = []

    def on_bar(
        self,
        bar: Dict[str, float],
        spread: Optional[float] = None,
        as_of: Optional[pd.Timestamp] = None,
    ) -> Dict[str, object]:
        t = pd.Timestamp(as_of if as_of is not None else bar.get("t"))
        close = bar.get("close")
        self.dc.update(t, close)
        feats = self.features.on_bar(bar)
        scored = self.tracker.score_step(feats)
        no_data = bool(scored.get("no_data"))
        regime = scored.get("regime", "NO DATA")
        p1 = scored.get("p_state1")
        events = self.windows.on_prob(t, regime, p1) if not no_data else []
        self._last_windows = events
        fast = self.fast_path.on_bar(bar, rv_20d=feats.get("rv_20d"), spread=spread)
        return {
            "as_of": t,
            "symbol": None,
            "regime": regime,
            "p_states": {"risk_on": scored.get("p_state0"), "risk_off": p1},
            "map_state": scored.get("map_state"),
            "classifier_version": self.classifier_version,
            "fast_flag": bool(fast["fast_flag"]),
            "fast_reasons": list(fast["fast_reasons"]),
            "fast_confirm_streak": fast["confirm_streak"],
            "fast_confirm_required": fast["confirm_required"],
            "stale": False,
            "no_data": no_data,
            "window_events": [
                {"start": str(w.start), "end": str(w.end) if w.end else None, "label": w.label}
                for w in events
            ],
        }
