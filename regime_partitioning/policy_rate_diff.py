import pandas as pd
from regime_partitioning.interest_rates import series_dict
from regime_partitioning.interest_rates import (
    us_monthly_interest_rates_df,
    australia_monthly_interest_rates_df,
    canada_monthly_interest_rates_df,
    japan_monthly_interest_rates_df,
    newzealand_monthly_interest_rates_df,
    us_quarterly_interest_rates_df,
    australia_quarterly_interest_rates_df,
    canada_quarterly_interest_rates_df,
    japan_quarterly_interest_rates_df,
    newzealand_quarterly_interest_rates_df,
    us_annual_interest_rates_df,
    australia_annual_interest_rates_df,
    canada_annual_interest_rates_df,
    japan_annual_interest_rates_df,
    newzealand_annual_interest_rates_df,
)

# ---- config ---------------------------------------------------------------

# Map FX pairs to (base_country, quote_country)
PAIR_COUNTRIES = {
    "AUDUSD": ("Australia", "United States"),
    "USDCAD": ("United States", "Canada"),
    "USDJPY": ("United States", "Japan"),
    "NZDUSD": ("New Zealand", "United States"),
    # common crosses
    "AUDJPY": ("Australia", "Japan"),
    "CADJPY": ("Canada", "Japan"),
    "AUDNZD": ("Australia", "New Zealand"),
    "AUDCAD": ("Australia", "Canada"),
    "NZDJPY": ("New Zealand", "Japan"),
    "CADUSD": ("Canada", "United States"),
}

FREQ_PRIORITY = ["Monthly", "Quarterly", "Annual"]  # highest → lowest

# ---- helpers --------------------------------------------------------------


def _pick_best_freq_df(series_dict, country):
    """Return highest-frequency DataFrame for a country."""
    for freq in FREQ_PRIORITY:
        key = (country, freq)
        if (
            key in series_dict
            and series_dict[key] is not None
            and len(series_dict[key]) > 0
        ):
            return series_dict[key].copy(), freq
    raise KeyError(
        f"No interest-rate series found for {country} at any supported frequency."
    )


def _normalize_ir_df(df, date_col=None, value_col=None, freq_hint=None):
    """
    Normalize interest-rate input (Series or DataFrame) to a monthly DatetimeIndex DataFrame with a single column 'rate'.
    - If input is a Series: index is dates, values are rates.
    - If input is a DataFrame: infer date/value columns if not provided.
    """
    # If already a Series of rates indexed by date
    if isinstance(df, pd.Series):
        out = df.to_frame(name="rate").copy()
        # ensure datetime index
        if not isinstance(out.index, pd.DatetimeIndex):
            out.index = pd.to_datetime(out.index)
        out = out.sort_index()
        # upsample to month-end and forward-fill
        out = out.asfreq("ME", method="ffill")
        return out

    # Infer columns if not provided
    if date_col is None:
        date_col = next(
            c for c in df.columns if "date" in c.lower() or "time" in c.lower()
        )
    if value_col is None:
        # pick the last numeric column
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if not num_cols:
            raise ValueError("No numeric rate column found.")
        value_col = num_cols[-1]

    out = df[[date_col, value_col]].dropna().copy()
    out[date_col] = pd.to_datetime(out[date_col])
    out = out.sort_values(date_col).set_index(date_col)
    out = out.rename(columns={value_col: "rate"})

    # upsample to month-end for uniformity
    # If input is monthly, this is a no-op. If quarterly/annual, expand and ffill.
    out = out.asfreq("ME", method="ffill")
    return out


def _policy_spread(series_dict, base_country, quote_country):
    """base − quote monthly policy-rate spread."""
    base_df, _ = _pick_best_freq_df(series_dict, base_country)
    quote_df, _ = _pick_best_freq_df(series_dict, quote_country)

    base_m = _normalize_ir_df(base_df)
    quote_m = _normalize_ir_df(quote_df)

    spread = base_m.join(
        quote_m, how="inner", lsuffix="_base", rsuffix="_quote"
    )
    spread["policy_rate_diff"] = spread["rate_base"] - spread["rate_quote"]
    return spread[["policy_rate_diff"]]


def _policy_spread_from_series(base_series: pd.Series, quote_series: pd.Series) -> pd.DataFrame:
    """Build spread DataFrame from two country Series (any freq), upsampled to monthly."""
    base_m = _normalize_ir_df(base_series)
    quote_m = _normalize_ir_df(quote_series)
    spread = base_m.join(quote_m, how="inner", lsuffix="_base", rsuffix="_quote")
    spread["policy_rate_diff"] = spread["rate_base"] - spread["rate_quote"]
    return spread[["policy_rate_diff"]]


def build_policy_spreads(series_dict, pair_countries=PAIR_COUNTRIES):
    """
    Returns dict: {pair: DataFrame(month-end, 'policy_rate_diff')}.
    """
    out = {}
    for pair, (base_ctry, quote_ctry) in pair_countries.items():
        out[pair] = _policy_spread(series_dict, base_ctry, quote_ctry)
    return out


# ---- build & exports ------------------------------------------------------

rate_diff_dict = build_policy_spreads(series_dict)

# Expose pair-specific DataFrames
audusd_policy_rate_diff = rate_diff_dict["AUDUSD"]
usdcad_policy_rate_diff = rate_diff_dict["USDCAD"]
usdjpy_policy_rate_diff = rate_diff_dict["USDJPY"]
nzdusd_policy_rate_diff = rate_diff_dict["NZDUSD"]
audjpy_policy_rate_diff = rate_diff_dict["AUDJPY"]
cadjpy_policy_rate_diff = rate_diff_dict["CADJPY"]
audnzd_policy_rate_diff = rate_diff_dict["AUDNZD"]
audcad_policy_rate_diff = rate_diff_dict["AUDCAD"]
nzdjpy_policy_rate_diff = rate_diff_dict["NZDJPY"]
cadusd_policy_rate_diff = rate_diff_dict["CADUSD"]

# Explicit frequency-based spreads (monthly index) using pre-split Series
# Monthly
audusd_policy_rate_diff_monthly = _policy_spread_from_series(
    australia_monthly_interest_rates_df, us_monthly_interest_rates_df
)
usdcad_policy_rate_diff_monthly = _policy_spread_from_series(
    us_monthly_interest_rates_df, canada_monthly_interest_rates_df
)
usdjpy_policy_rate_diff_monthly = _policy_spread_from_series(
    us_monthly_interest_rates_df, japan_monthly_interest_rates_df
)
nzdusd_policy_rate_diff_monthly = _policy_spread_from_series(
    newzealand_monthly_interest_rates_df, us_monthly_interest_rates_df
)
# Common crosses
audjpy_policy_rate_diff_monthly = _policy_spread_from_series(
    australia_monthly_interest_rates_df, japan_monthly_interest_rates_df
)
cadjpy_policy_rate_diff_monthly = _policy_spread_from_series(
    canada_monthly_interest_rates_df, japan_monthly_interest_rates_df
)
audnzd_policy_rate_diff_monthly = _policy_spread_from_series(
    australia_monthly_interest_rates_df, newzealand_monthly_interest_rates_df
)
audcad_policy_rate_diff_monthly = _policy_spread_from_series(
    australia_monthly_interest_rates_df, canada_monthly_interest_rates_df
)
nzdjpy_policy_rate_diff_monthly = _policy_spread_from_series(
    newzealand_monthly_interest_rates_df, japan_monthly_interest_rates_df
)
cadusd_policy_rate_diff_monthly = _policy_spread_from_series(
    canada_monthly_interest_rates_df, us_monthly_interest_rates_df
)

# Quarterly (upsampled to monthly)
audusd_policy_rate_diff_quarterly = _policy_spread_from_series(
    australia_quarterly_interest_rates_df, us_quarterly_interest_rates_df
)
usdcad_policy_rate_diff_quarterly = _policy_spread_from_series(
    us_quarterly_interest_rates_df, canada_quarterly_interest_rates_df
)
usdjpy_policy_rate_diff_quarterly = _policy_spread_from_series(
    us_quarterly_interest_rates_df, japan_quarterly_interest_rates_df
)
nzdusd_policy_rate_diff_quarterly = _policy_spread_from_series(
    newzealand_quarterly_interest_rates_df, us_quarterly_interest_rates_df
)
audjpy_policy_rate_diff_quarterly = _policy_spread_from_series(
    australia_quarterly_interest_rates_df, japan_quarterly_interest_rates_df
)
cadjpy_policy_rate_diff_quarterly = _policy_spread_from_series(
    canada_quarterly_interest_rates_df, japan_quarterly_interest_rates_df
)
audnzd_policy_rate_diff_quarterly = _policy_spread_from_series(
    australia_quarterly_interest_rates_df, newzealand_quarterly_interest_rates_df
)
audcad_policy_rate_diff_quarterly = _policy_spread_from_series(
    australia_quarterly_interest_rates_df, canada_quarterly_interest_rates_df
)
nzdjpy_policy_rate_diff_quarterly = _policy_spread_from_series(
    newzealand_quarterly_interest_rates_df, japan_quarterly_interest_rates_df
)
cadusd_policy_rate_diff_quarterly = _policy_spread_from_series(
    canada_quarterly_interest_rates_df, us_quarterly_interest_rates_df
)

# Annual (upsampled to monthly)
audusd_policy_rate_diff_annual = _policy_spread_from_series(
    australia_annual_interest_rates_df, us_annual_interest_rates_df
)
usdcad_policy_rate_diff_annual = _policy_spread_from_series(
    us_annual_interest_rates_df, canada_annual_interest_rates_df
)
usdjpy_policy_rate_diff_annual = _policy_spread_from_series(
    us_annual_interest_rates_df, japan_annual_interest_rates_df
)
nzdusd_policy_rate_diff_annual = _policy_spread_from_series(
    newzealand_annual_interest_rates_df, us_annual_interest_rates_df
)
audjpy_policy_rate_diff_annual = _policy_spread_from_series(
    australia_annual_interest_rates_df, japan_annual_interest_rates_df
)
cadjpy_policy_rate_diff_annual = _policy_spread_from_series(
    canada_annual_interest_rates_df, japan_annual_interest_rates_df
)
audnzd_policy_rate_diff_annual = _policy_spread_from_series(
    australia_annual_interest_rates_df, newzealand_annual_interest_rates_df
)
audcad_policy_rate_diff_annual = _policy_spread_from_series(
    australia_annual_interest_rates_df, canada_annual_interest_rates_df
)
nzdjpy_policy_rate_diff_annual = _policy_spread_from_series(
    newzealand_annual_interest_rates_df, japan_annual_interest_rates_df
)
cadusd_policy_rate_diff_annual = _policy_spread_from_series(
    canada_annual_interest_rates_df, us_annual_interest_rates_df
)

__all__ = [
    "build_policy_spreads",
    "rate_diff_dict",
    "audusd_policy_rate_diff",
    "usdcad_policy_rate_diff",
    "usdjpy_policy_rate_diff",
    "nzdusd_policy_rate_diff",
    "audjpy_policy_rate_diff",
    "cadjpy_policy_rate_diff",
    "audnzd_policy_rate_diff",
    "audcad_policy_rate_diff",
    "nzdjpy_policy_rate_diff",
    "cadusd_policy_rate_diff",
    # monthly
    "audusd_policy_rate_diff_monthly",
    "usdcad_policy_rate_diff_monthly",
    "usdjpy_policy_rate_diff_monthly",
    "nzdusd_policy_rate_diff_monthly",
    "audjpy_policy_rate_diff_monthly",
    "cadjpy_policy_rate_diff_monthly",
    "audnzd_policy_rate_diff_monthly",
    "audcad_policy_rate_diff_monthly",
    "nzdjpy_policy_rate_diff_monthly",
    "cadusd_policy_rate_diff_monthly",
    # quarterly
    "audusd_policy_rate_diff_quarterly",
    "usdcad_policy_rate_diff_quarterly",
    "usdjpy_policy_rate_diff_quarterly",
    "nzdusd_policy_rate_diff_quarterly",
    "audjpy_policy_rate_diff_quarterly",
    "cadjpy_policy_rate_diff_quarterly",
    "audnzd_policy_rate_diff_quarterly",
    "audcad_policy_rate_diff_quarterly",
    "nzdjpy_policy_rate_diff_quarterly",
    "cadusd_policy_rate_diff_quarterly",
    # annual
    "audusd_policy_rate_diff_annual",
    "usdcad_policy_rate_diff_annual",
    "usdjpy_policy_rate_diff_annual",
    "nzdusd_policy_rate_diff_annual",
    "audjpy_policy_rate_diff_annual",
    "cadjpy_policy_rate_diff_annual",
    "audnzd_policy_rate_diff_annual",
    "audcad_policy_rate_diff_annual",
    "nzdjpy_policy_rate_diff_annual",
    "cadusd_policy_rate_diff_annual",
]
