import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def clean_us_2y_yields():
    """
    Clean US 2Y yields data - already in good format
    """
    print("Processing US 2Y yields...")
    
    # Read the file
    file_path = "/home/matrillo/apps/regime-classification/data/yields/unclean/us_2y_yields.csv"
    df = pd.read_csv(file_path)
    
    # Check columns and data
    print(f"Columns: {df.columns.tolist()}")
    print(f"Date range: {df['observation_date'].min()} to {df['observation_date'].max()}")
    print(f"Data shape: {df.shape}")
    
    # Convert date column
    df['observation_date'] = pd.to_datetime(df['observation_date'])
    
    # Filter from 2000 onwards
    df_filtered = df[df['observation_date'] >= '2000-01-01'].copy()
    
    # Handle missing values (convert empty strings to NaN)
    df_filtered['DGS2'] = pd.to_numeric(df_filtered['DGS2'], errors='coerce')
    
    # Rename columns for consistency
    df_filtered = df_filtered.rename(columns={'observation_date': 'date', 'DGS2': '2y_yield'})
    
    print(f"Filtered data shape: {df_filtered.shape}")
    print(f"Missing values: {df_filtered['2y_yield'].isna().sum()}")
    
    return df_filtered

def clean_au_2y_yields():
    """
    Clean Australian 2Y yields data
    """
    print("\nProcessing Australian 2Y yields...")
    
    file_path = "/home/matrillo/apps/regime-classification/data/yields/unclean/au_yields.csv"
    df = pd.read_csv(file_path)
    
    print(f"Columns: {df.columns.tolist()}")
    print(f"Date range: {df['observation_date'].min()} to {df['observation_date'].max()}")
    print(f"Data shape: {df.shape}")
    
    # Convert date column
    df['observation_date'] = pd.to_datetime(df['observation_date'])
    
    # Filter from 2000 onwards
    df_filtered = df[df['observation_date'] >= '2000-01-01'].copy()
    
    # Check if AU_2Y column has any data
    if 'AU_2Y' in df_filtered.columns:
        # Handle missing values
        df_filtered['AU_2Y'] = pd.to_numeric(df_filtered['AU_2Y'], errors='coerce')
        
        # Check if we have any non-null 2Y data from 2000 onwards
        non_null_count = df_filtered['AU_2Y'].notna().sum()
        print(f"Non-null 2Y yield values from 2000 onwards: {non_null_count}")
        
        if non_null_count == 0:
            print("WARNING: No 2Y yield data available from 2000 onwards for Australia")
            return None
        
        # Rename columns for consistency
        df_filtered = df_filtered[['observation_date', 'AU_2Y']].rename(
            columns={'observation_date': 'date', 'AU_2Y': '2y_yield'})
        
        print(f"Filtered data shape: {df_filtered.shape}")
        print(f"Missing values: {df_filtered['2y_yield'].isna().sum()}")
        
        return df_filtered
    else:
        print("ERROR: AU_2Y column not found")
        return None

def clean_ca_2y_yields():
    """
    Clean Canadian 2Y yields data - complex format with metadata
    """
    print("\nProcessing Canadian 2Y yields...")
    
    file_path = "/home/matrillo/apps/regime-classification/data/yields/unclean/ca_yields.csv"
    
    # Find the header row (contains BD.CDN.2YR.DQ.YLD)
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    header_line = None
    for i, line in enumerate(lines):
        if 'BD.CDN.2YR.DQ.YLD' in line and 'date' in line:
            header_line = i
            break
    
    if header_line is None:
        print("ERROR: Could not find header line in Canadian yields file")
        return None
    
    print(f"Found header at line {header_line + 1}")
    
    # Read from header line onwards
    df = pd.read_csv(file_path, skiprows=header_line)
    
    print(f"Columns: {df.columns.tolist()}")
    print(f"Data shape: {df.shape}")
    
    # Convert date column
    df['date'] = pd.to_datetime(df['date'])
    
    # Filter from 2000 onwards
    df_filtered = df[df['date'] >= '2000-01-01'].copy()
    
    # Get 2Y yield column (BD.CDN.2YR.DQ.YLD)
    yield_col = 'BD.CDN.2YR.DQ.YLD'
    if yield_col not in df_filtered.columns:
        print(f"ERROR: {yield_col} column not found")
        return None
    
    # Handle missing values
    df_filtered[yield_col] = pd.to_numeric(df_filtered[yield_col], errors='coerce')
    
    # Select and rename columns
    df_filtered = df_filtered[['date', yield_col]].rename(
        columns={yield_col: '2y_yield'})
    
    print(f"Date range: {df_filtered['date'].min()} to {df_filtered['date'].max()}")
    print(f"Filtered data shape: {df_filtered.shape}")
    print(f"Missing values: {df_filtered['2y_yield'].isna().sum()}")
    
    return df_filtered

def clean_jpy_2y_yields():
    """
    Clean Japanese 2Y yields data
    """
    print("\nProcessing Japanese 2Y yields...")
    
    file_path = "/home/matrillo/apps/regime-classification/data/yields/unclean/jpy_yields.csv"
    
    # Read file, skipping the first row which contains units info
    df = pd.read_csv(file_path, skiprows=1)
    
    print(f"Columns: {df.columns.tolist()}")
    print(f"Data shape: {df.shape}")
    
    # Convert date column (format appears to be YYYY/M/D)
    df['Date'] = pd.to_datetime(df['Date'], format='%Y/%m/%d')
    
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    # Filter from 2000 onwards
    df_filtered = df[df['Date'] >= '2000-01-01'].copy()
    
    # Get 2Y yield column
    if '2Y' not in df_filtered.columns:
        print("ERROR: 2Y column not found")
        return None
    
    # Handle missing values
    df_filtered['2Y'] = pd.to_numeric(df_filtered['2Y'], errors='coerce')
    
    # Select and rename columns
    df_filtered = df_filtered[['Date', '2Y']].rename(
        columns={'Date': 'date', '2Y': '2y_yield'})
    
    print(f"Filtered data shape: {df_filtered.shape}")
    print(f"Missing values: {df_filtered['2y_yield'].isna().sum()}")
    
    return df_filtered

def clean_nz_2y_yields():
    """
    Clean New Zealand 2Y yields data - complex header structure
    """
    print("\nProcessing New Zealand 2Y yields...")
    
    file_path = "/home/matrillo/apps/regime-classification/data/yields/unclean/nz_yields.csv"
    
    # Read file, skip the first few rows which contain metadata
    # Look for row that contains "2 year" in the header
    df = pd.read_csv(file_path, skiprows=1)  # Skip first row
    
    print(f"Initial columns: {df.columns.tolist()[:10]}...")  # Show first 10 columns
    print(f"Data shape: {df.shape}")
    
    # Find the 2Y column - should be around index 8 based on earlier inspection
    # Look for a column that might contain "2 year" data
    header_row = df.iloc[0]  # Second row should contain the actual header names
    
    # Find column index for "2 year"
    two_year_col_idx = None
    for i, val in enumerate(header_row):
        if isinstance(val, str) and '2 year' in val:
            two_year_col_idx = i
            break
    
    if two_year_col_idx is None:
        print("WARNING: Could not find '2 year' column, trying column index 8...")
        two_year_col_idx = 8  # Based on earlier inspection
    
    print(f"Using column index {two_year_col_idx} for 2Y yields")
    
    # Skip metadata rows and read with proper header
    df = pd.read_csv(file_path, skiprows=4)  # Skip first 4 rows to get to data
    
    # Use the first column as date and the identified column as 2Y yield
    if df.shape[1] <= two_year_col_idx:
        print(f"ERROR: File only has {df.shape[1]} columns, can't access column {two_year_col_idx}")
        return None
    
    # Get column names
    date_col = df.columns[0]
    yield_col = df.columns[two_year_col_idx]
    
    print(f"Date column: {date_col}")
    print(f"Yield column: {yield_col}")
    
    # Convert date column
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    
    # Filter valid dates
    df = df.dropna(subset=[date_col])
    
    print(f"Date range: {df[date_col].min()} to {df[date_col].max()}")
    
    # Filter from 2000 onwards
    df_filtered = df[df[date_col] >= '2000-01-01'].copy()
    
    # Handle missing values for yield column
    df_filtered[yield_col] = pd.to_numeric(df_filtered[yield_col], errors='coerce')
    
    # Check if we have data from 2000 onwards
    non_null_count = df_filtered[yield_col].notna().sum()
    print(f"Non-null 2Y yield values from 2000 onwards: {non_null_count}")
    
    if non_null_count == 0:
        print("WARNING: No 2Y yield data available from 2000 onwards for New Zealand")
        return None
    
    # Select and rename columns
    df_filtered = df_filtered[[date_col, yield_col]].rename(
        columns={date_col: 'date', yield_col: '2y_yield'})
    
    print(f"Filtered data shape: {df_filtered.shape}")
    print(f"Missing values: {df_filtered['2y_yield'].isna().sum()}")
    
    return df_filtered

def clean_eu_2y_yields():
    """
    Clean European 2Y yields data - very complex ECB format
    """
    print("\nProcessing European 2Y yields...")
    
    file_path = "/home/matrillo/apps/regime-classification/data/yields/unclean/eu_yields.csv"
    
    print("WARNING: EU yields file is very large (3.9GB) and complex.")
    print("Attempting to find 2Y yield data...")
    
    # Try to find 2Y yield series in the ECB data
    # Look for patterns that might indicate 2Y yields
    try:    
        eu_yields = pd.read_csv(file_path)
        print("Found and loaded the data")
        eu_yields = eu_yields[["OBS_VALUE", "TIME_PERIOD"]]
        eu_yields.rename(columns={"TIME_PERIOD":"date","OBS_VALUE":"2y_yield"}, inplace=True)
        # convert to datetime
        eu_yields["date"] = pd.to_datetime(eu_yields["date"], errors="coerce")

        # filter from 2000 onwards
        eu_yields = eu_yields[eu_yields["date"] >= "2000-01-01"]

        # Remove rows with invalid dates
        eu_yields = eu_yields.dropna(subset=["date"]).sort_values("date")
        return eu_yields
        
    except Exception as e:
        print(f"Error processing EU yields: {e}")
        return None

def main():
    """
    Main function to clean all 2Y yield files and export to clean directory
    """
    print("=" * 60)
    print("CLEANING 2Y YIELD DATA")
    print("=" * 60)
    
    # Create output directory
    output_dir = Path("/home/matrillo/apps/regime-classification/data/yields/clean")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each country's data
    countries = {
        # 'us': clean_us_2y_yields,
        # 'au': clean_au_2y_yields, 
        # 'ca': clean_ca_2y_yields,
        # 'jpy': clean_jpy_2y_yields,
        # 'nz': clean_nz_2y_yields,
        'eu': clean_eu_2y_yields
    }
    
    results = {}
    
    for country, clean_func in countries.items():
        try:
            df = clean_func()
            if df is not None:
                # Export to CSV
                output_file = output_dir / f"{country}_2y_yields_clean.csv"
                df.to_csv(output_file, index=False)
                print(f"✓ Saved {country.upper()} data to {output_file}")
                results[country] = df
            else:
                print(f"✗ No data available for {country.upper()}")
                results[country] = None
        except Exception as e:
            print(f"✗ Error processing {country.upper()}: {e}")
            results[country] = None
        
        print("-" * 40)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for country, df in results.items():
        if df is not None:
            print(f"{country.upper()}: {len(df)} observations from {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
        else:
            print(f"{country.upper()}: No data available from 2000 onwards")
    
    return results

if __name__ == "__main__":
    results = main()
