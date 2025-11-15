from __future__ import annotations

from typing import List, NamedTuple, Optional

import numpy as np
import pandas as pd


class DCState:
    def __init__(self, theta_pct: float = 0.4):
        self.theta_pct = float(theta_pct)
        self.theta = self.theta_pct / 100.0
        self.initialized = False
        self.direction: Optional[int] = None  # +1 up, -1 down
        self.extreme_price: Optional[float] = None
        self.trend_start_price: Optional[float] = None
        self.trend_start_time: Optional[pd.Timestamp] = None
        self.bars_in_trend: int = 0
        self.tmv_accum: float = 0.0
        self.prev_price: Optional[float] = None


class DCEvent(NamedTuple):
    t: pd.Timestamp
    r: float
    tlen: int
    tmv: float


class DCUpdater:
    def __init__(self, theta_pct: float = 0.4):
        self.state = DCState(theta_pct)

    def _init_if_needed(self, t: pd.Timestamp, price: float):
        if not self.state.initialized:
            self.state.initialized = True
            self.state.extreme_price = price
            self.state.trend_start_price = price
            self.state.trend_start_time = pd.Timestamp(t)
            self.state.direction = None
            self.state.prev_price = price
            self.state.bars_in_trend = 0
            self.state.tmv_accum = 0.0

    def update(self, t: pd.Timestamp, price: float) -> List[DCEvent]:
        t = pd.Timestamp(t)
        self._init_if_needed(t, float(price))
        s = self.state

        events: List[DCEvent] = []

        # update TMV accumulator with absolute log-return each bar
        if s.prev_price is not None and s.prev_price > 0:
            s.tmv_accum += abs(np.log(price / s.prev_price))
        s.prev_price = float(price)
        s.bars_in_trend += 1

        # If direction not set yet, wait for first theta move to set direction
        if s.direction is None:
            up_trigger = price >= s.extreme_price * (1.0 + s.theta)
            down_trigger = price <= s.extreme_price * (1.0 - s.theta)
            if up_trigger:
                s.direction = +1
                s.trend_start_price = s.extreme_price
                s.trend_start_time = t
                s.extreme_price = price
                s.bars_in_trend = 1
                s.tmv_accum = 0.0
            elif down_trigger:
                s.direction = -1
                s.trend_start_price = s.extreme_price
                s.trend_start_time = t
                s.extreme_price = price
                s.bars_in_trend = 1
                s.tmv_accum = 0.0
            return events

        # Update extreme within current direction
        if s.direction == +1:
            if price > s.extreme_price:
                s.extreme_price = price
        elif s.direction == -1:
            if price < s.extreme_price:
                s.extreme_price = price

        # Check for reversal (directional change)
        if s.direction == +1:
            reversal = price <= s.extreme_price * (1.0 - s.theta)
        else:
            reversal = price >= s.extreme_price * (1.0 + s.theta)

        if reversal:
            r = (
                float(np.log(s.extreme_price / s.trend_start_price))
                if s.direction == +1
                else float(np.log(s.trend_start_price / s.extreme_price))
            )
            events.append(
                DCEvent(
                    t=s.trend_start_time if s.trend_start_time is not None else t,
                    r=r,
                    tlen=s.bars_in_trend,
                    tmv=float(s.tmv_accum),
                )
            )

            # Start new trend in opposite direction
            s.direction = -s.direction
            s.trend_start_price = s.extreme_price
            s.trend_start_time = t
            s.extreme_price = price
            s.bars_in_trend = 1
            s.tmv_accum = 0.0
            s.prev_price = float(price)

        return events
