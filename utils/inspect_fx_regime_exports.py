# %% [markdown]
"""
Interactive inspection of FX regime-labelled OHLCV CSV exports.

This file is intended to be used in an interactive Python environment
(VS Code / Jupyter "Python: Interactive" / IPython) with cell support.
"""

# %%
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


# %% [markdown]
"""
## Configuration

- Assumes regime-labelled CSVs are under `exports/forex/`.
- Each CSV is named `{SYMBOL}_regime_ohlcv.csv` as produced by `fx_regime_dataset_export.py`.
"""

# %%
PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPORT_DIR = PROJECT_ROOT / "exports" / "forex"

# Hard-code a few symbols to inspect
INSPECT_SYMBOLS: List[str] = [
    "EURUSD",
    "USDJPY",
    "EURJPY",
]


# %%
def load_regime_csv(symbol: str) -> pd.DataFrame:
    """Load a regime-labelled OHLCV CSV for a given symbol."""
    path = EXPORT_DIR / f"{symbol}_regime_ohlcv.csv"
    if not path.exists():
        raise FileNotFoundError(f"CSV not found for symbol {symbol}: {path}")

    df = pd.read_csv(path, parse_dates=["datetime"])
    df = df.set_index("datetime").sort_index()
    return df


# %%
REGIME_DF: Dict[str, pd.DataFrame] = {}
for sym in INSPECT_SYMBOLS:
    try:
        REGIME_DF[sym] = load_regime_csv(sym)
    except FileNotFoundError as e:
        print(e)


# %% [markdown]
"""
## Plotting utilities

`plot_regime_candles(symbol, start_date, end_date)`

- Shows a simple candlestick plot for the selected date range.
- Shades background regions where `final_regime` is constant.
- Prints the set of regimes present in the selected window.
"""


# %%
def _get_regime_palette(regimes: List[str]) -> Dict[str, str]:
    """Assign a stable color to each regime label."""
    # Use a matplotlib colormap for a limited number of regimes.
    cmap = plt.get_cmap("tab20")
    colors: Dict[str, str] = {}
    for i, reg in enumerate(sorted(regimes)):
        colors[reg] = cmap(i % cmap.N)
    return colors


# %%
def plot_regime_candles(
    symbol: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> None:
    """Plot candlesticks with regime shading for a given symbol and date range.

    Parameters
    ----------
    symbol : str
        FX symbol, e.g. "EURUSD".
    start_date : str, optional
        Start date (inclusive), e.g. "2020-01-01".
    end_date : str, optional
        End date (inclusive), e.g. "2020-06-30".
    """
    if symbol not in REGIME_DF:
        raise KeyError(
            f"Symbol {symbol} not loaded. Loaded symbols: {sorted(REGIME_DF.keys())}"
        )

    df = REGIME_DF[symbol]

    # Slice by date range
    if start_date is not None:
        df = df.loc[start_date:]
    if end_date is not None:
        df = df.loc[:end_date]

    if df.empty:
        print(f"No data for {symbol} in the given date range.")
        return

    # Print regimes present in the filtered window
    macro_states = (
        sorted(df["macro_state"].dropna().unique().tolist())
        if "macro_state" in df.columns
        else []
    )
    vol_states = (
        sorted(df["vol_state"].dropna().unique().tolist())
        if "vol_state" in df.columns
        else []
    )
    final_regimes = (
        sorted(df["final_regime"].dropna().unique().tolist())
        if "final_regime" in df.columns
        else []
    )

    print(f"Symbol: {symbol}")
    print(f"Date range: {df.index.min().date()} -> {df.index.max().date()}")
    print("Macro states in window:", macro_states)
    print("Vol states in window:", vol_states)
    print("Final regimes in window:", final_regimes)

    # Prepare price data
    price_cols = ["open", "high", "low", "close"]
    missing_cols = [c for c in price_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing OHLC columns for {symbol}: {missing_cols}")

    px = df[price_cols].dropna()
    if px.empty:
        print("No OHLC data to plot.")
        return

    # Convert dates to matplotlib's numeric format
    x = mdates.date2num(px.index.to_pydatetime())

    fig, ax = plt.subplots(figsize=(12, 6))

    # Simple candlestick drawing: wick + thick body line
    for xi, (idx, row) in zip(x, px.iterrows()):
        o, h, l, c = row["open"], row["high"], row["low"], row["close"]
        color = "green" if c >= o else "red"
        # Wick
        ax.vlines(xi, l, h, color=color, linewidth=1)
        # Body (thicker line between open and close)
        ax.vlines(xi, o, c, color=color, linewidth=4)

    # Regime shading based on final_regime
    if "final_regime" in df.columns:
        reg_series = df["final_regime"].fillna("unknown")
        unique_regimes = sorted(reg_series.unique().tolist())
        palette = _get_regime_palette(unique_regimes)

        current_reg = None
        segment_start = None

        for ts, reg in reg_series.items():
            if current_reg is None:
                current_reg = reg
                segment_start = ts
            elif reg != current_reg:
                # Shade previous segment
                ax.axvspan(
                    segment_start, ts, color=palette.get(current_reg, "0.9"), alpha=0.12
                )
                current_reg = reg
                segment_start = ts

        # Shade last segment
        if current_reg is not None and segment_start is not None:
            ax.axvspan(
                segment_start,
                reg_series.index[-1],
                color=palette.get(current_reg, "0.9"),
                alpha=0.12,
            )

    ax.set_title(f"{symbol} with regimes ({start_date} to {end_date})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()

    plt.tight_layout()
    plt.show()


# %% [markdown]
"""
## Example usage

Uncomment and run a cell like this to visually inspect a symbol:

```python
plot_regime_candles("EURJPY", "2020-01-01", "2020-06-30")
```
"""

# %%
# Example (commented out by default)
# plot_regime_candles("EURJPY", "2020-01-01", "2020-06-30")
