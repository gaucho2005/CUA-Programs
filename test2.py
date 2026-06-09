import polars as pl
from pathlib import Path

# Path to your file
file_path = Path("/mnt/c/Users/ciroj/Desktop/Parquet Files/TnP/MultifileHistograms/AllFilesData/DataE_2022.parquet")

def check_file_structure(path):
    # Scan the file to get metadata without loading data
    lf = pl.scan_parquet(path)
    schema = lf.schema
    
    print(f"Checking file: {path.name}")
    print(f"Total columns found: {len(schema)}")
    
    # 1. Check for nesting
    # A flat table should have simple types (Float, Int, Boolean, etc.)
    # If it contains 'List' or 'Struct', it is likely nested incorrectly
    non_flat_cols = [name for name, dtype in schema.items() if isinstance(dtype, (pl.List, pl.Struct))]
    
    if non_flat_cols:
        print(f"WARNING: Found nested columns: {non_flat_cols}")
        print("This file may be a 'list of lists' rather than a flat table.")
    else:
        print("SUCCESS: All 118 columns are flat. The data is properly structured for tabular analysis.")

    # 2. Verify row count
    # We collect just the length to see if the file is readable as a single entity
    row_count = lf.select(pl.len()).collect().item()
    print(f"Total rows in file: {row_count}")

check_file_structure(file_path)