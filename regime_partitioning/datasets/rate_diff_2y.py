import pandas as pd


def rate_diff_2y(df_home: pd.DataFrame, df_foreign: pd.DataFrame) -> pd.Series:
    """
    Compute 2Y yield differential: home - foreign (BASE - QUOTE).
    Expects each DataFrame to have columns ['date', '2y_yield'].
    """
    # Remove duplicates and handle missing values before setting index
    df_home_clean = df_home.dropna(subset=['date', '2y_yield']).drop_duplicates(subset=['date']).copy()
    df_foreign_clean = df_foreign.dropna(subset=['date', '2y_yield']).drop_duplicates(subset=['date']).copy()
    
    s_home = df_home_clean.set_index("date")["2y_yield"]
    s_foreign = df_foreign_clean.set_index("date")["2y_yield"]
    diff = (s_home - s_foreign).dropna()
    diff.name = "rate_diff_2y"
    return diff

# Load cleaned 2Y yield data (assumed correct and standardized as ['date','2y_yield'])
usd_2y_yield = pd.read_csv("data/yields/clean/us_2y_yields_clean.csv", parse_dates=["date"]).sort_values("date")
eur_2y_yield = pd.read_csv("data/yields/clean/eu_2y_yields_clean.csv", parse_dates=["date"]).sort_values("date")
jpy_2y_yield = pd.read_csv("data/yields/clean/jpy_2y_yields_clean.csv", parse_dates=["date"]).sort_values("date")
aus_2y_yield = pd.read_csv("data/yields/clean/au_2y_yields_clean.csv", parse_dates=["date"]).sort_values("date")
nz_2y_yield = pd.read_csv("data/yields/clean/nz_2y_yields_clean.csv", parse_dates=["date"]).sort_values("date")

# Explicit rate differential Series for available FX majors/minors (BASE - QUOTE)
# Majors
EURUSD_rate_diff_2y = rate_diff_2y(eur_2y_yield, usd_2y_yield)
USDJPY_rate_diff_2y = rate_diff_2y(usd_2y_yield, jpy_2y_yield)
AUDUSD_rate_diff_2y = rate_diff_2y(aus_2y_yield, usd_2y_yield)
NZDUSD_rate_diff_2y = rate_diff_2y(nz_2y_yield, usd_2y_yield)

# Minors
EURJPY_rate_diff_2y = rate_diff_2y(eur_2y_yield, jpy_2y_yield)
EURAUD_rate_diff_2y = rate_diff_2y(eur_2y_yield, aus_2y_yield)
EURNZD_rate_diff_2y = rate_diff_2y(eur_2y_yield, nz_2y_yield)
AUDJPY_rate_diff_2y = rate_diff_2y(aus_2y_yield, jpy_2y_yield)
NZDJPY_rate_diff_2y = rate_diff_2y(nz_2y_yield, jpy_2y_yield)
AUDNZD_rate_diff_2y = rate_diff_2y(aus_2y_yield, nz_2y_yield)

# Combined DataFrame with fixed, explicit columns
rate_diff_2y_df = (
    pd.concat(
        [
            EURUSD_rate_diff_2y.rename("EURUSD_rate_diff_2y"),
            USDJPY_rate_diff_2y.rename("USDJPY_rate_diff_2y"),
            AUDUSD_rate_diff_2y.rename("AUDUSD_rate_diff_2y"),
            NZDUSD_rate_diff_2y.rename("NZDUSD_rate_diff_2y"),
            EURJPY_rate_diff_2y.rename("EURJPY_rate_diff_2y"),
            EURAUD_rate_diff_2y.rename("EURAUD_rate_diff_2y"),
            EURNZD_rate_diff_2y.rename("EURNZD_rate_diff_2y"),
            AUDJPY_rate_diff_2y.rename("AUDJPY_rate_diff_2y"),
            NZDJPY_rate_diff_2y.rename("NZDJPY_rate_diff_2y"),
            AUDNZD_rate_diff_2y.rename("AUDNZD_rate_diff_2y"),
        ],
        axis=1,
    )
    .sort_index()
)
