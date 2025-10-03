import pandas as pd
from datetime import datetime
from pathlib import Path
import requests

# Majors excluding EUR (you said EUR is covered separately)
# Try multiple country code variants for problematic countries
ISO3_VARIANTS = {
    "USA": ["USA"],
    "JPN": ["JPN"],
    "GBR": ["GBR"],
    "CAN": ["CAN"],
    "AUS": ["AUS", "AU", "AUSTRALIA"],  # Try variants
    "NZL": ["NZL", "NZ", "NEW_ZEALAND"],  # Try variants
    "CHE": ["CHE", "CH", "SWITZERLAND"],
}

# DB.NOMICS series key pattern (YoY, all items less food & energy)
BASE_2018 = (
    "https://api.db.nomics.world/v22/series/OECD/"
    "DSD_PRICES_COICOP2018@DF_PRICES_C2018_N_TXCP01_NRG/"
    "{c}.M.N.CPI.PA._TXCP01_NRG.N.GY.csv"
)
BASE_1999 = (
    "https://api.db.nomics.world/v22/series/OECD/"
    "DSD_PRICES@DF_PRICES_N_TXCP01_NRG/"
    "{c}.M.N.CPI.PA._TXCP01_NRG.N.GY.csv"
)

# Fallback: headline CPI (all items) if core CPI not available
HEADLINE_2018 = (
    "https://api.db.nomics.world/v22/series/OECD/"
    "DSD_PRICES_COICOP2018@DF_PRICES_C2018_ALL/"
    "{c}.M.N.CPI.PA.ALL.N.GY.csv"
)
HEADLINE_1999 = (
    "https://api.db.nomics.world/v22/series/OECD/"
    "DSD_PRICES@DF_PRICES_ALL/"
    "{c}.M.N.CPI.PA.ALL.N.GY.csv"
)

outdir = Path("core_cpi_yoy_COICOP")
outdir.mkdir(parents=True, exist_ok=True)


def check_url_exists(url):
    """Check if URL returns valid data"""
    try:
        response = requests.head(url)
        return response.status_code == 200
    except:
        return False


all_urls = []
frames = {}
for country_name, variants in ISO3_VARIANTS.items():
    print(f"Processing {country_name}...")

    df = None
    src = None
    used_url = None
    used_code = None

    # Try each variant
    for code in variants:
        url18 = BASE_2018.format(c=code)
        url99 = BASE_1999.format(c=code)
        headline18 = HEADLINE_2018.format(c=code)
        headline99 = HEADLINE_1999.format(c=code)

        # try core CPI first (2018, then 1999)
        for url_type, url, cpi_type in [
            ("2018 Core", url18, "core"),
            ("1999 Core", url99, "core"),
            ("2018 Headline", headline18, "headline"),
            ("1999 Headline", headline99, "headline"),
        ]:
            try:
                if check_url_exists(url):
                    df = pd.read_csv(url)
                    src = url_type
                    used_url = url
                    used_code = code
                    print(f"  ✓ Using {url_type} CPI for {country_name} (code: {code})")
                    break
                else:
                    continue
            except Exception as e:
                print(f"  - {url_type} failed for code {code}: {e}")
                continue

        if df is not None:
            break

    if df is None:
        print(f"  ✗ All codes failed for {country_name}")
        continue

    if df.empty:
        print(f"  ✗ No data for {country_name}")
        continue

    # Get the value column (it's the second column, first is period)
    value_col = df.columns[1]
    df = df[["period", value_col]].copy()

    # Filter to data starting FROM 2010 (not ending at 2010)
    df["period"] = pd.to_datetime(df["period"])
    df = df[df["period"] >= "2010-01-01"].copy()

    if df.empty:
        print(f"  ⚠ No data from 2010+ for {country_name}")
        continue

    # Convert period back to string for consistency
    df["period"] = df["period"].dt.strftime("%Y-%m")
    df.rename(columns={value_col: country_name}, inplace=True)

    # Save individual country file
    df.to_csv(outdir / f"{country_name}_core_cpi_yoy_{src}_from_2010.csv", index=False)

    frames[country_name] = df.set_index("period")
    all_urls.append((country_name, src, used_url, used_code))
    print(
        f"  ✓ Saved {len(df)} records for {country_name} from {df['period'].iloc[0]} to {df['period'].iloc[-1]}"
    )

if frames:
    # combined wide file (aligned index)
    wide = pd.concat(frames.values(), axis=1)
    wide.to_csv(outdir / "G7xEUR_core_cpi_yoy_from_2010_wide.csv")
    print(
        f"\n✓ Combined file saved with {len(wide)} rows and {len(wide.columns)} countries"
    )
else:
    print("\n✗ No data successfully downloaded")

# print the exact URLs used
print("\nURLs used:")
for country, src, url, code in all_urls:
    print(f"{country} [{src}] (code: {code}): {url}")

print(f"\nFiles saved to: {outdir.absolute()}")
