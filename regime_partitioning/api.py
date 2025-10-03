# Economic Data API Specifications
# INTEREST RATES //
# IMF ONLY SUPPORTS MONTHLY.
# ECB ONLY SUPPORTS MONTHLY.

# CPI //
# IMF SUPPORTS ANNUAL, MONTHLY, QUARTERLY.
# ECB ONLY SUPPORTS MONTHLY.

# UNEMPLOYMENT //
# IMF SUPPORTS ANNUAL, MONTHLY, QUARTERLY.
# ECB ONLY SUPPORTS QUARTERLY.

# GDP //
# IMF SUPPORTS YEARLY, QUARTERLY.
# ECB ONLY SUPPORTS QUARTERLY.

# BOP //
# IMF SUPPORTS YEARLY, QUARTERLY

BASE_URL = "http://192.168.1.212:8000"

import requests
from typing import Optional, Tuple, Dict, Any


def fetch_economic_data(currency: str, data_type: str, granularity: str) -> Optional[pd.DataFrame]:
    """
    Fetch economic data from the API endpoint.
    
    Args:
        currency: Currency code (e.g., 'usd', 'eur')
        data_type: Type of data ('cpi', 'interest_rates', 'gdp', 'unemployment')
        granularity: Time granularity ('monthly', 'quarterly', 'annual')
    
    Returns:
        DataFrame with economic data or None if request fails
    """
    url = f"{BASE_URL}/api/economics/{currency}/{data_type}/{granularity}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        # Convert to DataFrame assuming the API returns data in a structured format
        df = pd.DataFrame(data)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
        
        return df
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {data_type} data for {currency}: {e}")
        return None
