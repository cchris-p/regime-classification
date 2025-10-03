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
