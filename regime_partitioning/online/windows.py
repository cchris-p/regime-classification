"""Debounced regime-window logic with hysteresis and minimum dwell."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd


@dataclass
class WindowRule:
    open_p: float = 0.80
    close_p: float = 0.50
    confirm_open: int = 2
    confirm_close: int = 2
    min_dwell_bars: int = 2


@dataclass
class Window:
    start: pd.Timestamp
    end: Optional[pd.Timestamp]
    label: str
    meta: dict = field(default_factory=dict)


class WindowStateMachine:
    """Emit regime-window open/close events without single-bar flips.

    A flip requires ``confirm`` consecutive bars on the new label, the matching
    probability threshold, and ``min_dwell_bars`` since the last flip.
    """

    def __init__(self, rule: Optional[WindowRule] = None) -> None:
        self.rule = rule or WindowRule()
        self.current: Optional[Window] = None
        self.label: Optional[str] = None
        self._streak = 0
        self._bars_in_state = 0

    def _threshold_ok(self, target: str, p_regime2: float) -> bool:
        if target == "risk_on":
            return (1.0 - p_regime2) >= self.rule.open_p
        return p_regime2 >= self.rule.close_p

    def on_prob(
        self, t: pd.Timestamp, regime: str, p_regime2: Optional[float]
    ) -> List[Window]:
        events: List[Window] = []
        if p_regime2 is None or regime not in ("risk_on", "risk_off"):
            return events
        if self.label is None:
            self.label = regime
            self.current = Window(start=t, end=None, label=regime)
            self._bars_in_state = 0
            events.append(self.current)
            return events

        if regime == self.label:
            self._streak = 0
            self._bars_in_state += 1
            return events

        if not self._threshold_ok(regime, p_regime2):
            self._streak = 0
            self._bars_in_state += 1
            return events

        self._streak += 1
        self._bars_in_state += 1
        confirm = (
            self.rule.confirm_open if regime == "risk_on" else self.rule.confirm_close
        )
        if self._streak >= confirm and self._bars_in_state >= self.rule.min_dwell_bars:
            closed = Window(start=self.current.start, end=t, label=self.label)
            events.append(closed)
            self.label = regime
            self.current = Window(start=t, end=None, label=regime)
            self._streak = 0
            self._bars_in_state = 0
        return events
