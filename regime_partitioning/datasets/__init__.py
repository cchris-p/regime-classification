import pandas as pd
import numpy as np
from trading_utils.get_forex_data import get_forex_data_by_pair
from .rate_diff_2y import (
    rate_diff_2y_df,
    EURUSD_rate_diff_2y,
    USDJPY_rate_diff_2y,
    AUDUSD_rate_diff_2y,
    NZDUSD_rate_diff_2y,
    EURJPY_rate_diff_2y,
    EURAUD_rate_diff_2y,
    EURNZD_rate_diff_2y,
    AUDJPY_rate_diff_2y,
    NZDJPY_rate_diff_2y,
    AUDNZD_rate_diff_2y,
)
from .cpi_diff_core import (
    cpi_diff_core_daily_df,
    EURUSD_cpi_diff_core_daily,
    USDJPY_cpi_diff_core_daily,
    AUDUSD_cpi_diff_core_daily,
    NZDUSD_cpi_diff_core_daily,
    EURJPY_cpi_diff_core_daily,
    EURAUD_cpi_diff_core_daily,
    EURNZD_cpi_diff_core_daily,
    AUDJPY_cpi_diff_core_daily,
    NZDJPY_cpi_diff_core_daily,
    AUDNZD_cpi_diff_core_daily,
)
from .rv_20d import rv_20d


def create_fx_dataset_for_pair(symbol, start_date="2020-01-01", end_date="2024-12-31"):
    """
    Create forex dataset for a specific pair.
    """
    # Get forex data and log transform daily closes
    df_fx = get_forex_data_by_pair(
        symbol=symbol, start_date=start_date, end_date=end_date, granularity="D"
    )

    if "close" not in df_fx.columns:
        raise ValueError(f"'close' column not found in data for {symbol}")

    # Log returns from log-transformed closes
    df_fx = df_fx.sort_index()
    df_fx["ret"] = np.log(df_fx["close"] / df_fx["close"].shift(1))

    # 20-day realized volatility
    df_fx["rv_20d"] = df_fx["ret"].rolling(20).std() * np.sqrt(252)  # annualized

    return df_fx[["ret", "rv_20d"]].dropna()


# Hard variables for each FX cross with complete feature set
# df_fx: indexed by date, must contain at least:
#   'ret'  -> FX returns (daily log/close-to-close)
#   'rv_20d' -> realized volatility (rolling 20d stdev of returns)
#   'rate_diff_2y' -> 2y yield differential (home - foreign)

# EURUSD
EURUSD_fx = create_fx_dataset_for_pair("EURUSD")
EURUSD_fx = EURUSD_fx.join(EURUSD_rate_diff_2y.rename("rate_diff_2y"), how="left")
EURUSD_fx = EURUSD_fx.join(
    EURUSD_cpi_diff_core_daily.rename("cpi_diff_core"), how="left"
)
EURUSD_dataset = {"symbol": "EURUSD", "df_fx": EURUSD_fx.dropna()}

# USDJPY
USDJPY_fx = create_fx_dataset_for_pair("USDJPY")
USDJPY_fx = USDJPY_fx.join(USDJPY_rate_diff_2y.rename("rate_diff_2y"), how="left")
USDJPY_fx = USDJPY_fx.join(
    USDJPY_cpi_diff_core_daily.rename("cpi_diff_core"), how="left"
)
USDJPY_dataset = {"symbol": "USDJPY", "df_fx": USDJPY_fx.dropna()}

# AUDUSD
AUDUSD_fx = create_fx_dataset_for_pair("AUDUSD")
AUDUSD_fx = AUDUSD_fx.join(AUDUSD_rate_diff_2y.rename("rate_diff_2y"), how="left")
AUDUSD_fx = AUDUSD_fx.join(
    AUDUSD_cpi_diff_core_daily.rename("cpi_diff_core"), how="left"
)
AUDUSD_dataset = {"symbol": "AUDUSD", "df_fx": AUDUSD_fx.dropna()}

# NZDUSD
NZDUSD_fx = create_fx_dataset_for_pair("NZDUSD")
NZDUSD_fx = NZDUSD_fx.join(NZDUSD_rate_diff_2y.rename("rate_diff_2y"), how="left")
NZDUSD_fx = NZDUSD_fx.join(
    NZDUSD_cpi_diff_core_daily.rename("cpi_diff_core"), how="left"
)
NZDUSD_dataset = {"symbol": "NZDUSD", "df_fx": NZDUSD_fx.dropna()}

# EURJPY
EURJPY_fx = create_fx_dataset_for_pair("EURJPY")
EURJPY_fx = EURJPY_fx.join(EURJPY_rate_diff_2y.rename("rate_diff_2y"), how="left")
EURJPY_fx = EURJPY_fx.join(
    EURJPY_cpi_diff_core_daily.rename("cpi_diff_core"), how="left"
)
EURJPY_dataset = {"symbol": "EURJPY", "df_fx": EURJPY_fx.dropna()}

# EURAUD
EURAUD_fx = create_fx_dataset_for_pair("EURAUD")
EURAUD_fx = EURAUD_fx.join(EURAUD_rate_diff_2y.rename("rate_diff_2y"), how="left")
EURAUD_fx = EURAUD_fx.join(
    EURAUD_cpi_diff_core_daily.rename("cpi_diff_core"), how="left"
)
EURAUD_dataset = {"symbol": "EURAUD", "df_fx": EURAUD_fx.dropna()}

# EURNZD
EURNZD_fx = create_fx_dataset_for_pair("EURNZD")
EURNZD_fx = EURNZD_fx.join(EURNZD_rate_diff_2y.rename("rate_diff_2y"), how="left")
EURNZD_fx = EURNZD_fx.join(
    EURNZD_cpi_diff_core_daily.rename("cpi_diff_core"), how="left"
)
EURNZD_dataset = {"symbol": "EURNZD", "df_fx": EURNZD_fx.dropna()}

# AUDJPY
AUDJPY_fx = create_fx_dataset_for_pair("AUDJPY")
AUDJPY_fx = AUDJPY_fx.join(AUDJPY_rate_diff_2y.rename("rate_diff_2y"), how="left")
AUDJPY_fx = AUDJPY_fx.join(
    AUDJPY_cpi_diff_core_daily.rename("cpi_diff_core"), how="left"
)
AUDJPY_dataset = {"symbol": "AUDJPY", "df_fx": AUDJPY_fx.dropna()}

# NZDJPY
NZDJPY_fx = create_fx_dataset_for_pair("NZDJPY")
NZDJPY_fx = NZDJPY_fx.join(NZDJPY_rate_diff_2y.rename("rate_diff_2y"), how="left")
NZDJPY_fx = NZDJPY_fx.join(
    NZDJPY_cpi_diff_core_daily.rename("cpi_diff_core"), how="left"
)
NZDJPY_dataset = {"symbol": "NZDJPY", "df_fx": NZDJPY_fx.dropna()}

# AUDNZD
AUDNZD_fx = create_fx_dataset_for_pair("AUDNZD")
AUDNZD_fx = AUDNZD_fx.join(AUDNZD_rate_diff_2y.rename("rate_diff_2y"), how="left")
AUDNZD_fx = AUDNZD_fx.join(
    AUDNZD_cpi_diff_core_daily.rename("cpi_diff_core"), how="left"
)
AUDNZD_dataset = {"symbol": "AUDNZD", "df_fx": AUDNZD_fx.dropna()}

# Dictionary of all available datasets
fx_datasets = {
    "EURUSD": EURUSD_dataset,
    "USDJPY": USDJPY_dataset,
    "AUDUSD": AUDUSD_dataset,
    "NZDUSD": NZDUSD_dataset,
    "EURJPY": EURJPY_dataset,
    "EURAUD": EURAUD_dataset,
    "EURNZD": EURNZD_dataset,
    "AUDJPY": AUDJPY_dataset,
    "NZDJPY": NZDJPY_dataset,
    "AUDNZD": AUDNZD_dataset,
}

# Default export (EURUSD for backward compatibility)
df_fx = EURUSD_dataset["df_fx"]
