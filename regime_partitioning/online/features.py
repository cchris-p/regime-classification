"""Per-bar feature construction for the online regime tracker."""

from __future__ import annotations

import math
from collections import deque
from typing import Deque, Dict, Optional

import numpy as np
import pandas as pd


class FeatureBuilder:
    """Maintain ``ret`` and ``rv_20d`` from a stream of daily bars.

    ``rv_20d`` matches the offline definition: rolling 20-observation standard
    deviation of daily log returns, annualized by ``sqrt(252)``.
    """

    def __init__(self, rv_window: int = 20, ann: int = 252) -> None:
        self.rv_window = int(rv_window)
        self.ann = int(ann)
        self._closes: Deque[float] = deque(maxlen=self.rv_window + 2)
        self._last_close: Optional[float] = None

    def reset(self) -> None:
        self._closes.clear()
        self._last_close = None

    def on_bar(self, bar: Dict[str, float]) -> pd.Series:
        close = bar.get("close")
        if close is None or pd.isna(close) or close <= 0:
            return pd.Series({"ret": np.nan, "rv_20d": np.nan})
        close = float(close)
        if self._last_close is None or self._last_close <= 0:
            ret = np.nan
        else:
            ret = math.log(close / self._last_close)
        self._last_close = close
        if not pd.isna(ret):
            self._closes.append(ret)
        else:
            self._closes.append(0.0)
        if len(self._closes) >= self.rv_window:
            window = np.asarray(list(self._closes)[-self.rv_window :], dtype=float)
            rv = float(np.std(window, ddof=1) * math.sqrt(self.ann))
        else:
            rv = np.nan
        return pd.Series({"ret": ret, "rv_20d": rv})

    def warmup(self) -> bool:
        return len(self._closes) >= self.rv_window
