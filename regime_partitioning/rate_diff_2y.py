"""
2-Year Government Yield Rate Differentials

This module implements the preferred driver for the minimal regime classification plan,
using 2-year government yield spreads as the rate_diff_2y feature.

Output index is month-end and intended to be forward-filled to daily before model merges.
Policy-rate spread (from policy_rate_diff.py) serves as a fallback when 2-year yields 
are unavailable.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from regime_partitioning.policy_rate_diff import PAIR_COUNTRIES

# ---- Data Loading & Country Series ----------------------------------------

def load_2y_yield_data() -> Dict[str, pd.Series]:
    """
    Load 2-year government yield data for all countries.
    
    NOTE: This is a placeholder function. Replace with actual data sources
    such as FRED, Bloomberg, Refinitiv APIs, or CSV files.
    
    Returns:
        Dict mapping country names to 2-year yield Series
    """
    # TODO: Replace with actual data loading
    # Example structure for when real data is available:
    
    countries = ["United States", "Australia", "Canada", "Japan", "New Zealand"]
    yield_data = {}
    
    for country in countries:
        # Placeholder: Create sample data structure
        # In real implementation, this would be:
        # yield_data[country] = load_from_fred(country_2y_code)
        # or: yield_data[country] = pd.read_csv(f"{country}_2y_yields.csv")
        
        # For now, create empty series with proper structure
        yield_data[country] = pd.Series(dtype=float, name="yield_2y")
        
    return yield_data


def _normalize_2y_yield(data: pd.Series, country: str) -> pd.DataFrame:
    """
    Normalize 2-year yield data to monthly month-end frequency.
    
    Normalization contract:
    - Convert index to DatetimeIndex, sort ascending
    - Resample/upsample to month-end and forward-fill
    - Keep units as provided (percent or decimal) but ensure consistency
    - Rename numeric column to 'yield_2y'
    - Return single-column DataFrame
    
    Args:
        data: Raw 2-year yield Series
        country: Country name for error reporting
        
    Returns:
        DataFrame with DatetimeIndex and 'yield_2y' column
    """
    if data.empty:
        raise ValueError(f"No 2-year yield data available for {country}")
    
    # Ensure DatetimeIndex
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)
    
    # Sort ascending
    data = data.sort_index()
    
    # Convert to DataFrame and rename column
    df = data.to_frame(name="yield_2y")
    
    # Resample to month-end and forward-fill
    df = df.asfreq("ME", method="ffill")
    
    return df


def _check_unit_consistency(base_df: pd.DataFrame, quote_df: pd.DataFrame, 
                           base_country: str, quote_country: str) -> None:
    """
    Check that both countries use consistent units (both % or both decimal).
    
    Raises ValueError if units appear to be inconsistent.
    """
    # Simple heuristic: if typical values are >10, assume percentage; if <1, assume decimal
    base_mean = base_df["yield_2y"].dropna().mean()
    quote_mean = quote_df["yield_2y"].dropna().mean()
    
    base_is_percent = base_mean > 1.0
    quote_is_percent = quote_mean > 1.0
    
    if base_is_percent != quote_is_percent:
        raise ValueError(
            f"Unit mismatch between {base_country} (avg: {base_mean:.3f}) and "
            f"{quote_country} (avg: {quote_mean:.3f}). "
            f"Both should use same units (percentage or decimal). "
            f"Standardize before computing spreads."
        )


# ---- Core Functions -------------------------------------------------------

def _build_2y_spread(base_country: str, quote_country: str, 
                     yield_data: Dict[str, pd.Series]) -> pd.DataFrame:
    """
    Build 2-year yield spread: base - quote.
    
    Args:
        base_country: Base currency country
        quote_country: Quote currency country  
        yield_data: Dict of country -> 2Y yield Series
        
    Returns:
        DataFrame with 'rate_diff_2y' column
        
    Raises:
        KeyError: If country data is missing
        ValueError: If unit consistency check fails
    """
    # Check data availability
    if base_country not in yield_data:
        raise KeyError(f"2-year yield data missing for {base_country}")
    if quote_country not in yield_data:
        raise KeyError(f"2-year yield data missing for {quote_country}")
    
    # Normalize both series
    base_df = _normalize_2y_yield(yield_data[base_country], base_country)
    quote_df = _normalize_2y_yield(yield_data[quote_country], quote_country)
    
    # Check unit consistency
    if not base_df.empty and not quote_df.empty:
        _check_unit_consistency(base_df, quote_df, base_country, quote_country)
    
    # Join on date index (inner join to avoid NaN issues)
    spread = base_df.join(quote_df, how="inner", lsuffix="_base", rsuffix="_quote")
    
    # Calculate spread: base - quote
    spread["rate_diff_2y"] = spread["yield_2y_base"] - spread["yield_2y_quote"]
    
    return spread[["rate_diff_2y"]]


def build_2y_spreads(yield_data: Dict[str, pd.Series], 
                     pair_countries: Dict[str, Tuple[str, str]] = PAIR_COUNTRIES) -> Dict[str, pd.DataFrame]:
    """
    Build 2-year yield spreads for all currency pairs.
    
    Args:
        yield_data: Dict mapping country names to 2Y yield Series
        pair_countries: Dict mapping FX tickers to (base_country, quote_country)
        
    Returns:
        Dict mapping FX tickers to rate_diff_2y DataFrames
    """
    spreads = {}
    
    for pair, (base_country, quote_country) in pair_countries.items():
        try:
            spreads[pair] = _build_2y_spread(base_country, quote_country, yield_data)
        except (KeyError, ValueError) as e:
            print(f"Warning: Could not build 2Y spread for {pair}: {e}")
            # Create empty DataFrame with proper structure
            spreads[pair] = pd.DataFrame(columns=["rate_diff_2y"])
    
    return spreads


def _resample_to_daily(monthly_spread: pd.DataFrame, 
                      business_day_index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Helper to reindex monthly spread to business daily frequency with forward-fill.
    
    Args:
        monthly_spread: Monthly spread DataFrame
        business_day_index: Target business day DatetimeIndex
        
    Returns:
        Daily spread DataFrame
    """
    return monthly_spread.reindex(business_day_index, method="ffill")


# ---- Data Loading & Country Variables -------------------------------------

# Load 2-year yield data (replace with actual data sources)
_yield_data = load_2y_yield_data()

# Create normalized country-specific DataFrames
# TODO: Replace load_2y_yield_data() with actual data loading
try:
    us_2y_df = _normalize_2y_yield(_yield_data["United States"], "United States")
except (KeyError, ValueError):
    us_2y_df = pd.DataFrame(columns=["yield_2y"])

try:
    australia_2y_df = _normalize_2y_yield(_yield_data["Australia"], "Australia")
except (KeyError, ValueError):
    australia_2y_df = pd.DataFrame(columns=["yield_2y"])

try:
    canada_2y_df = _normalize_2y_yield(_yield_data["Canada"], "Canada")
except (KeyError, ValueError):
    canada_2y_df = pd.DataFrame(columns=["yield_2y"])

try:
    japan_2y_df = _normalize_2y_yield(_yield_data["Japan"], "Japan")
except (KeyError, ValueError):
    japan_2y_df = pd.DataFrame(columns=["yield_2y"])

try:
    newzealand_2y_df = _normalize_2y_yield(_yield_data["New Zealand"], "New Zealand")
except (KeyError, ValueError):
    newzealand_2y_df = pd.DataFrame(columns=["yield_2y"])


# ---- Build & Export Spreads -----------------------------------------------

# Build all 2-year yield spreads
rate_diff_2y_dict = build_2y_spreads(_yield_data)

# Export individual major pair spreads
audusd_rate_diff_2y = rate_diff_2y_dict.get("AUDUSD", pd.DataFrame(columns=["rate_diff_2y"]))
usdcad_rate_diff_2y = rate_diff_2y_dict.get("USDCAD", pd.DataFrame(columns=["rate_diff_2y"]))
usdjpy_rate_diff_2y = rate_diff_2y_dict.get("USDJPY", pd.DataFrame(columns=["rate_diff_2y"]))
nzdusd_rate_diff_2y = rate_diff_2y_dict.get("NZDUSD", pd.DataFrame(columns=["rate_diff_2y"]))

# Export common cross pair spreads
audjpy_rate_diff_2y = rate_diff_2y_dict.get("AUDJPY", pd.DataFrame(columns=["rate_diff_2y"]))
cadjpy_rate_diff_2y = rate_diff_2y_dict.get("CADJPY", pd.DataFrame(columns=["rate_diff_2y"]))
audnzd_rate_diff_2y = rate_diff_2y_dict.get("AUDNZD", pd.DataFrame(columns=["rate_diff_2y"]))
audcad_rate_diff_2y = rate_diff_2y_dict.get("AUDCAD", pd.DataFrame(columns=["rate_diff_2y"]))
nzdjpy_rate_diff_2y = rate_diff_2y_dict.get("NZDJPY", pd.DataFrame(columns=["rate_diff_2y"]))
cadusd_rate_diff_2y = rate_diff_2y_dict.get("CADUSD", pd.DataFrame(columns=["rate_diff_2y"]))


# ---- Public API -----------------------------------------------------------

__all__ = [
    # Core functions
    "build_2y_spreads",
    "load_2y_yield_data",
    
    # Country DataFrames
    "us_2y_df",
    "australia_2y_df", 
    "canada_2y_df",
    "japan_2y_df",
    "newzealand_2y_df",
    
    # Spread dictionary
    "rate_diff_2y_dict",
    
    # Major pair spreads
    "audusd_rate_diff_2y",
    "usdcad_rate_diff_2y", 
    "usdjpy_rate_diff_2y",
    "nzdusd_rate_diff_2y",
    
    # Cross pair spreads
    "audjpy_rate_diff_2y",
    "cadjpy_rate_diff_2y",
    "audnzd_rate_diff_2y", 
    "audcad_rate_diff_2y",
    "nzdjpy_rate_diff_2y",
    "cadusd_rate_diff_2y",
]


# ---- Usage Example --------------------------------------------------------

def example_usage():
    """
    Example demonstrating how to use 2-year yield spreads.
    """
    print("=== 2-Year Yield Rate Differentials ===")
    
    print(f"Available pairs: {list(rate_diff_2y_dict.keys())}")
    
    # Example: USDJPY 2Y spread
    if not usdjpy_rate_diff_2y.empty:
        print(f"\nUSDJPY 2Y spread shape: {usdjpy_rate_diff_2y.shape}")
        print(f"Date range: {usdjpy_rate_diff_2y.index.min()} to {usdjpy_rate_diff_2y.index.max()}")
        print(f"Sample data:\n{usdjpy_rate_diff_2y.head()}")
    else:
        print("\nUSDJPY 2Y spread: No data available")
        print("TODO: Connect actual 2-year yield data sources")
    
    # Example: Daily resampling
    # business_days = pd.date_range("2020-01-01", "2024-12-31", freq="B")
    # daily_spread = _resample_to_daily(usdjpy_rate_diff_2y, business_days)


if __name__ == "__main__":
    example_usage()