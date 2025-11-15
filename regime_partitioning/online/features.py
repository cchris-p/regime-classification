from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np
import pandas as pd


class FeatureBuilder:
    def __init__(
        self, symbol: str = "EURUSD", rv_fallback: bool = True, rv_window: int = 20
    ):
        self.symbol = symbol
        self.rv_fallback = rv_fallback
        self.rv_window = int(rv_window)
        self._prev_close: Optional[float] = None
        self._rets = deque(maxlen=self.rv_window)

    def on_bar(self, bar: dict) -> pd.Series:
        t = pd.Timestamp(bar.get("t")) if bar.get("t") is not None else pd.NaT
        close = float(bar.get("close")) if bar.get("close") is not None else np.nan

        ret = np.nan
        if self._prev_close is not None and np.isfinite(close) and close > 0:
            ret = float(np.log(close / self._prev_close))
        if np.isfinite(ret):
            self._rets.append(ret)
        self._prev_close = close if np.isfinite(close) else self._prev_close

        rv_20d = np.nan
        if self.rv_fallback and len(self._rets) >= max(2, self.rv_window // 2):
            # simple rolling std; annualization factor for daily assumed
            rv_20d = float(np.std(self._rets, ddof=0)) * np.sqrt(252.0)

        return pd.Series({"ret": ret, "rv_20d": rv_20d}, name=t)
