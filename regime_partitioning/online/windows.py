from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import pandas as pd


@dataclass
class WindowRule:
    open_p: float = 0.80
    close_p: float = 0.50
    confirm_open: int = 2
    confirm_close: int = 2
    min_trends: int = 2


@dataclass
class Window:
    start: pd.Timestamp
    end: Optional[pd.Timestamp]
    label: str


class WindowStateMachine:
    def __init__(self, rule: WindowRule):
        self.rule = rule
        self.current: Optional[Window] = None
        self._open_streak = 0
        self._close_streak = 0
        self._trend_count = 0
        self._pending_open_time: Optional[pd.Timestamp] = None

    def reset(self):
        self.current = None
        self._open_streak = 0
        self._close_streak = 0
        self._trend_count = 0
        self._pending_open_time = None

    def on_prob(
        self, t: pd.Timestamp, p_regime2: float, dc_event: bool
    ) -> List[Window]:
        if not isinstance(t, pd.Timestamp):
            t = pd.Timestamp(t)

        changed: List[Window] = []

        # No active window: consider opening
        if self.current is None:
            if dc_event:
                if p_regime2 >= self.rule.open_p:
                    if self._open_streak == 0:
                        self._pending_open_time = t
                    self._open_streak += 1
                else:
                    self._open_streak = 0
                    self._pending_open_time = None

                if self._open_streak >= self.rule.confirm_open:
                    start_time = (
                        self._pending_open_time
                        if self._pending_open_time is not None
                        else t
                    )
                    self.current = Window(start=start_time, end=None, label="regime_2")
                    changed.append(self.current)
                    self._open_streak = 0
                    self._close_streak = 0
                    self._trend_count = 0
                    self._pending_open_time = None
            # nothing to close when no active window
            return changed

        # Active window exists: consider closing
        if dc_event:
            self._trend_count += 1
            if p_regime2 <= self.rule.close_p:
                self._close_streak += 1
            else:
                self._close_streak = 0

            if (
                self._close_streak >= self.rule.confirm_close
                and self._trend_count >= self.rule.min_trends
            ):
                closed = Window(
                    start=self.current.start, end=t, label=self.current.label
                )
                changed.append(closed)
                # reset state (no active window)
                self.current = None
                self._close_streak = 0
                self._open_streak = 0
                self._trend_count = 0
                self._pending_open_time = None

        return changed
