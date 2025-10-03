import pandas as pd
import numpy as np
from trading_utils.get_forex_data import get_forex_data_by_pair


def rv_20d(symbol, start_date, end_date, ann=True):
    """
    Compute 20-day realized volatility from daily close prices.
    Uses get_forex_data_by_pair() which already returns a DataFrame indexed by datetime.
    """
    df = get_forex_data_by_pair(symbol=symbol, start_date=start_date, end_date=end_date, granularity="D")

    if "close" not in df.columns:
        raise ValueError(f"'close' column not found in data for {symbol}")

    # use the existing DatetimeIndex
    s = df["close"].sort_index()

    # log returns
    rets = np.log(s / s.shift(1))

    # rolling 20-day stdev
    vol = rets.rolling(20).std()

    if ann:
        vol *= np.sqrt(252)

    vol.name = "rv_20d"
    return vol.dropna()