from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.patches import Patch
import pandas as pd
from bokeh.io import output_notebook, show
from bokeh.models import BoxAnnotation, DatetimeTickFormatter
from bokeh.plotting import figure


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPORT_DIR = PROJECT_ROOT / "exports" / "forex"

# Hard-code a few symbols to inspect
INSPECT_SYMBOLS: List[str] = [
    "EURUSD",
    "USDJPY",
    "EURJPY",
]


def load_regime_csv(symbol: str) -> pd.DataFrame:
    """Load a regime-labelled OHLCV CSV for a given symbol."""
    path = EXPORT_DIR / f"{symbol}_regime_ohlcv.csv"
    if not path.exists():
        raise FileNotFoundError(f"CSV not found for symbol {symbol}: {path}")

    df = pd.read_csv(path, parse_dates=["datetime"])
    df = df.set_index("datetime").sort_index()
    return df


REGIME_DF: Dict[str, pd.DataFrame] = {}
for sym in INSPECT_SYMBOLS:
    try:
        REGIME_DF[sym] = load_regime_csv(sym)
    except FileNotFoundError as e:
        print(e)


def _get_regime_palette(regimes: List[str]) -> Dict[str, str]:
    """Assign a stable color to each regime label."""
    # Use a matplotlib colormap, but convert to hex strings so colors
    # are valid for both Matplotlib and Bokeh.
    cmap = plt.get_cmap("tab20")
    palette: Dict[str, str] = {}
    for i, reg in enumerate(sorted(regimes)):
        rgba = cmap(i % cmap.N)
        palette[reg] = mcolors.to_hex(rgba)
    return palette


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

        # Legend mapping regime label -> shading color
        handles = [
            Patch(
                facecolor=palette.get(reg, "0.9"),
                edgecolor="none",
                alpha=0.4,
                label=reg,
            )
            for reg in unique_regimes
        ]
        if handles:
            ax.legend(
                handles=handles,
                title="Final regimes",
                loc="upper left",
                bbox_to_anchor=(1.01, 1.0),
                borderaxespad=0.0,
            )

    ax.set_title(f"{symbol} with regimes ({start_date} to {end_date})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()

    plt.tight_layout()
    plt.show()


def plot_regime_candles_bokeh(
    symbol: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> None:
    """Interactive candlestick plot with regime shading using Bokeh.

    Parameters
    ----------
    symbol : str
        FX symbol, e.g. "EURUSD".
    start_date : str, optional
        Start date (inclusive), e.g. "2020-01-01".
    end_date : str, optional
        End date (inclusive), e.g. "2021-06-30".
    """

    # Ensure Bokeh renders in the notebook
    output_notebook()

    if symbol in REGIME_DF:
        df = REGIME_DF[symbol]
    else:
        df = load_regime_csv(symbol)

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

    price_cols = ["open", "high", "low", "close"]
    missing_cols = [c for c in price_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing OHLC columns for {symbol}: {missing_cols}")

    px = df[price_cols].dropna()
    if px.empty:
        print("No OHLC data to plot.")
        return

    # Bokeh's ColumnDataSource will reset the index, so avoid having both
    # an index named 'datetime' and a 'datetime' column at the same time.
    df_plot = px.copy().reset_index()
    # After reset_index, the datetime column will typically be named 'datetime'.
    # If not, rename the first column defensively.
    if "datetime" not in df_plot.columns:
        first_col = df_plot.columns[0]
        df_plot = df_plot.rename(columns={first_col: "datetime"})
    df_plot["up"] = df_plot["close"] >= df_plot["open"]
    df_plot["color"] = np.where(df_plot["up"], "green", "red")

    # Candle width as ~70% of median bar spacing
    if len(df_plot) > 1:
        dt_diff = df_plot["datetime"].diff().median()
        if pd.isna(dt_diff):
            width_ms = 12 * 60 * 60 * 1000  # 12 hours
        else:
            width_ms = dt_diff.total_seconds() * 1000 * 0.7
    else:
        width_ms = 12 * 60 * 60 * 1000

    p = figure(
        x_axis_type="datetime",
        width=950,
        height=450,
        title=f"{symbol} with regimes ({start_date} to {end_date})",
        tools="pan,wheel_zoom,box_zoom,reset,save",
        active_drag="pan",
        active_scroll="wheel_zoom",
    )

    # Add candlesticks
    p.segment(
        x0="datetime",
        y0="high",
        x1="datetime",
        y1="low",
        color="color",
        source=df_plot,
    )
    p.vbar(
        x="datetime",
        width=width_ms,
        top="open",
        bottom="close",
        fill_color="color",
        line_color="color",
        source=df_plot,
    )

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
                box = BoxAnnotation(
                    left=segment_start,
                    right=ts,
                    fill_color=palette.get(current_reg, "lightgray"),
                    fill_alpha=0.12,
                    line_alpha=0.0,
                )
                p.add_layout(box)
                current_reg = reg
                segment_start = ts

        if current_reg is not None and segment_start is not None:
            box = BoxAnnotation(
                left=segment_start,
                right=reg_series.index[-1],
                fill_color=palette.get(current_reg, "lightgray"),
                fill_alpha=0.12,
                line_alpha=0.0,
            )
            p.add_layout(box)

        # Simple legend using small markers for each regime color
        y_min = px["low"].min()
        y_max = px["high"].max()
        y_span = y_max - y_min if y_max > y_min else 1.0
        x_ref = df_plot["datetime"].min()

        for i, reg in enumerate(unique_regimes):
            p.circle(
                x=[x_ref],
                y=[y_max + (i + 1) * 0.02 * y_span],
                size=8,
                color=palette.get(reg, "lightgray"),
                alpha=0.9,
                legend_label=reg,
            )

        p.legend.title = "Final regimes"
        p.legend.location = "top_left"
        p.legend.click_policy = "hide"

    p.xaxis.formatter = DatetimeTickFormatter(days="%Y-%m-%d")
    p.xaxis.major_label_orientation = 0.8

    show(p)
