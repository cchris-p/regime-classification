"""Online directional-change (DC) intrinsic-time state machine.

The offline pipeline uses PELT changepoints and a walk-forward HMM. The online
layer reuses the book's DC representation (R/T/TMV) as the intrinsic-time event
clock so the tracker is not driven by the arbitrary bar grid.

This is a deliberately small percentage-threshold DC implementation:

- track the latest extreme price and the direction of travel;
- emit a ``DCEvent`` when price reverses by ``theta_pct`` from that extreme;
- reset the extreme to the new reversal price.

It does not claim to reproduce every DC variant in the reference material; the
contract (completed events with return, bar length, total move value) is what
the window logic consumes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, NamedTuple, Optional

import pandas as pd


class DCEvent(NamedTuple):
    t: pd.Timestamp
    r: float
    tlen: int
    tmv: float
    direction: int


@dataclass
class DCState:
    theta_pct: float = 0.4
    extreme: Optional[float] = None
    direction: int = 0
    extreme_index: int = 0
    bars: int = 0
    start_price: Optional[float] = None


class DCUpdater:
    """Feed prices; receive completed directional-change events."""

    def __init__(self, theta_pct: float = 0.4) -> None:
        if theta_pct <= 0:
            raise ValueError("theta_pct must be positive")
        self.state = DCState(theta_pct=float(theta_pct))

    def update(self, t: pd.Timestamp, price: float) -> List[DCEvent]:
        st = self.state
        st.bars += 1
        if price is None or pd.isna(price) or price <= 0:
            return []
        if st.extreme is None:
            st.extreme = float(price)
            st.start_price = float(price)
            st.extreme_index = st.bars
            return []

        price = float(price)
        if st.direction == 0:
            move = (price - st.extreme) / st.extreme * 100.0
            if abs(move) >= st.theta_pct:
                st.direction = 1 if move > 0 else -1
                st.start_price = st.extreme
                st.extreme = price
                st.extreme_index = st.bars
            return []

        if st.direction == 1:
            if price >= st.extreme:
                st.extreme = price
                st.extreme_index = st.bars
                return []
            move = (price - st.extreme) / st.extreme * 100.0
            if move <= -st.theta_pct:
                start = st.start_price if st.start_price is not None else st.extreme
                event = DCEvent(
                    t=t,
                    r=(st.extreme - start) / start if start else 0.0,
                    tlen=max(1, st.bars - st.extreme_index),
                    tmv=(st.extreme - start) / start if start else 0.0,
                    direction=1,
                )
                st.direction = -1
                st.start_price = st.extreme
                st.extreme = price
                st.extreme_index = st.bars
                return [event]
            return []

        if price <= st.extreme:
            st.extreme = price
            st.extreme_index = st.bars
            return []
        move = (price - st.extreme) / st.extreme * 100.0
        if move >= st.theta_pct:
            start = st.start_price if st.start_price is not None else st.extreme
            event = DCEvent(
                t=t,
                r=(st.extreme - start) / start if start else 0.0,
                tlen=max(1, st.bars - st.extreme_index),
                tmv=(st.extreme - start) / start if start else 0.0,
                direction=-1,
            )
            st.direction = 1
            st.start_price = st.extreme
            st.extreme = price
            st.extreme_index = st.bars
            return [event]
        return []
