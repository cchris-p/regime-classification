"""Fast-path shift detector.

The slow HMM/macro label lags by construction. This detector flags a change in
market *conditions* within the session, independent of the slow regime label.

Resolved thresholds (board card `REGIME-002`, 2026-11-12):

- ``rv_20d > 2x`` its trailing 6-month (126-session) median;
- ADR usage ``> 100%`` of the trailing 20-session ADR;
- session gap ``> 1x`` ADR;
- spread ``> 2x`` the trailing 20-session median.

Any trigger must persist for ``2 bars or 1 hour, whichever is longer`` before
the fast flag is raised.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Deque, Dict, List, Optional

import numpy as np
import pandas as pd


def confirmation_bars(bar_minutes: Optional[float]) -> int:
    if not bar_minutes or bar_minutes <= 0:
        return 2
    return max(2, int(math.ceil(60.0 / float(bar_minutes))))


class FastPathDetector:
    def __init__(
        self,
        rv_spike_mult: float = 2.0,
        rv_lookback: int = 126,
        adr_lookback: int = 20,
        adr_usage_max: float = 1.0,
        gap_adr_mult: float = 1.0,
        spread_mult: float = 2.0,
        spread_lookback: int = 20,
        bar_minutes: Optional[float] = None,
    ) -> None:
        self.rv_spike_mult = float(rv_spike_mult)
        self.adr_lookback = int(adr_lookback)
        self.adr_usage_max = float(adr_usage_max)
        self.gap_adr_mult = float(gap_adr_mult)
        self.spread_mult = float(spread_mult)
        self.spread_lookback = int(spread_lookback)
        self.confirm = confirmation_bars(bar_minutes)

        self._rv: Deque[float] = deque(maxlen=int(rv_lookback))
        self._adr_pct: Deque[float] = deque(maxlen=self.adr_lookback)
        self._spread: Deque[float] = deque(maxlen=self.spread_lookback)
        self._prev_close: Optional[float] = None
        self._trigger_streak = 0
        self._last_reasons: List[str] = []
        self.fast_flag = False

    def on_bar(
        self,
        bar: Dict[str, float],
        rv_20d: Optional[float] = None,
        spread: Optional[float] = None,
    ) -> Dict[str, object]:
        reasons: List[str] = []
        high, low, close = bar.get("high"), bar.get("low"), bar.get("close")
        open_ = bar.get("open")

        rv = rv_20d if rv_20d is not None else bar.get("rv_20d")
        if rv is not None and not pd.isna(rv):
            rv = float(rv)
            if len(self._rv) >= 30:
                med = float(np.median(self._rv))
                if med > 0 and rv > self.rv_spike_mult * med:
                    reasons.append(f"rv_20d>{self.rv_spike_mult:g}x_6m_median")
            self._rv.append(rv)

        day_pct = None
        if all(v is not None and not pd.isna(v) for v in (high, low, close, open_)):
            high, low, close, open_ = map(float, (high, low, close, open_))
            day_pct = (high - low) / close if close else None
            if day_pct is not None and self._adr_pct:
                adr = float(np.median(self._adr_pct))
                if adr > 0:
                    usage = day_pct / adr
                    if usage > self.adr_usage_max:
                        reasons.append(f"adr_usage>{self.adr_usage_max:g}")
                    if self._prev_close and self._prev_close > 0:
                        gap = abs(open_ - self._prev_close) / self._prev_close
                        if gap > self.gap_adr_mult * adr:
                            reasons.append(f"gap>{self.gap_adr_mult:g}x_adr")
            self._prev_close = close
            if day_pct is not None:
                self._adr_pct.append(day_pct)

        if spread is not None and not pd.isna(spread):
            spread = float(spread)
            if len(self._spread) >= 10:
                med_s = float(np.median(self._spread))
                if med_s > 0 and spread > self.spread_mult * med_s:
                    reasons.append(f"spread>{self.spread_mult:g}x_20s_median")
            self._spread.append(spread)

        if reasons:
            self._trigger_streak += 1
            self._last_reasons = reasons
        else:
            self._trigger_streak = 0

        previously = self.fast_flag
        self.fast_flag = self._trigger_streak >= self.confirm
        return {
            "fast_flag": self.fast_flag,
            "triggered": bool(reasons),
            "confirm_streak": self._trigger_streak,
            "confirm_required": self.confirm,
            "fast_reasons": list(self._last_reasons) if self.fast_flag else reasons,
            "just_changed": self.fast_flag != previously,
        }
