from regime_partitioning.constants import forex_majors_countries
from regime_partitioning.constants import interest_rate_df_drop_columns
import pandas as pd
import re


def load_interest_rates_df():
    df = pd.read_csv(
        "data/dataset_2025-10-03T01_22_51.521947779Z_DEFAULT_INTEGRATION_IMF.STA_MFS_IR_8.0.1.csv"
    )
    df.drop(interest_rate_df_drop_columns, axis=1, inplace=True)
    countries = [i for i in set(df["COUNTRY"]) if i in forex_majors_countries]
    df = df[df["COUNTRY"].isin(countries)]
    return df

def parse_period(label: str) -> pd.Timestamp:
    """Convert column labels like 1950, 1950-Q1, 1950-M01 to a datetime."""
    if re.fullmatch(r"\d{4}", label):
        # Annual -> January of that year
        return pd.to_datetime(label + "-01-01")
    if re.fullmatch(r"\d{4}-Q[1-4]", label):
        # Quarterly -> use Period
        return pd.Period(label, freq="Q").to_timestamp(how="end")
    if re.fullmatch(r"\d{4}-M\d{2}", label):
        # Monthly -> YYYY-MM
        year, month = label.split("-M")
        return pd.to_datetime(f"{year}-{month}-01")
    return pd.NaT


def split_country_frequency(df: pd.DataFrame) -> dict:
    time_cols = [c for c in df.columns if c not in ["COUNTRY", "FREQUENCY"]]
    parsed_index = [parse_period(c) for c in time_cols]

    result = {}
    for _, row in df.iterrows():
        key = (row["COUNTRY"], row["FREQUENCY"])
        values = row[time_cols].values
        s = pd.Series(values, index=parsed_index).dropna()
        result[key] = s
    return result

df = load_interest_rates_df()
series_dict = split_country_frequency(df)

us_monthly_interest_rates_df = series_dict[("United States", "Monthly")]
australia_monthly_interest_rates_df = series_dict[("Australia", "Monthly")]
canada_monthly_interest_rates_df = series_dict[("Canada", "Monthly")]
japan_monthly_interest_rates_df = series_dict[("Japan", "Monthly")]
newzealand_monthly_interest_rates_df = series_dict[("New Zealand", "Monthly")]

us_quarterly_interest_rates_df = series_dict[("United States", "Quarterly")]
australia_quarterly_interest_rates_df = series_dict[("Australia", "Quarterly")]
canada_quarterly_interest_rates_df = series_dict[("Canada", "Quarterly")]
japan_quarterly_interest_rates_df = series_dict[("Japan", "Quarterly")]
newzealand_quarterly_interest_rates_df = series_dict[("New Zealand", "Quarterly")]

us_annual_interest_rates_df = series_dict[("United States", "Annual")]
australia_annual_interest_rates_df = series_dict[("Australia", "Annual")]
canada_annual_interest_rates_df = series_dict[("Canada", "Annual")]
japan_annual_interest_rates_df = series_dict[("Japan", "Annual")]
newzealand_annual_interest_rates_df = series_dict[("New Zealand", "Annual")]

