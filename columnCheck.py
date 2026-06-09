#for checking one column at a time
import pandas as pd
import vector
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import math
pd.set_option("display.max_rows", None)
from pathlib import Path
from tqdm import tqdm
import pyarrow.parquet as pq
skippedtotal=0
def skip_column(col):
    # skip booleans
    if col.dtype == bool:
        return True

    # skip columns of less than 1000 unique values
    if col.nunique() <= 1000:
        return True

    return False


out_folder = Path("/mnt/c/Users/ciroj/Desktop/Parquet Files/TnP/DY_postEE_2022/nominal/Columns/histograms")
folder = Path("/mnt/c/Users/ciroj/Desktop/Parquet Files/TnP/DY_postEE_2022/nominal/Columns")
parquet_files = list(folder.glob("*.parquet"))
totalnans=0

for f in tqdm(parquet_files,desc="makinghistograms"):

    pf = pq.ParquetFile(f)
    title=(pf.schema.names[0]) #this plus line before makes for much faster reading of names, maybe use that for something later
    
    if title !="probe_hoe":
        continue
    
    df=pd.read_parquet(f)
    col = df.iloc[:,0]
    """
    sorted_col= col.sort_values(ascending=False)
    print (sorted_col)
    print(col.nlargest(10).values) #we'll use this is we need speed   
    """
    
    """
    name= f.stem
    print(name)
    if skip_column(col):
        print("skipping invalid column (contains booleans or too few unique values)")
        skippedtotal+=1
        continue
    
    histmax=col.max()
    histmin=col.min()
    print("max=",histmax)
    print("min=",histmin)
    """
 
    
    filename = f"{f.stem}_hist3.png"
    
    nan_count = np.isnan(col).sum()
    print(f"Found and removed {nan_count} NaN values.")
    total=totalnans+nan_count
    col = col[np.isfinite(col)] #finds and removes nans
    
    counts, bin_edges = np.histogram(col, bins=100)
    
    if counts.max()>1100000: #implements different rules for histograms that would have an excessive amount of data concentrated in one bin
        counts, bin_edges = np.histogram(col, bins=100)
        
        indices = np.argsort(-counts) #if the highest bar is more than double the second highest, cut it off in the chart
        sorted_values = counts[indices]
        if sorted_values[0]>(2*sorted_values[1]): 
            plt.ylim(top=(1.1*sorted_values[1]),bottom=0)

    """
    if counts.max()>1100000:
        counts, bin_edges = np.histogram(col, bins=100)
        
    """
    mask = counts >= 1000
    filtered_counts = counts[mask] #filters out bars with less than a certain nummber of entries
    #print(f"Max bar is {filtered_counts.max()}")
    filtered_edges = bin_edges[:-1][mask]
    width = np.diff(bin_edges)[0]
    plt.bar(filtered_edges, filtered_counts, width=width, color='black')
    #plt.figure()
    #plt.hist(col, bins=100, range=(histmin,histmax),color='red')
    plt.ticklabel_format(style='plain', axis='y')
    plt.savefig(out_folder / filename)
    plt.close()
    
print(f"found total {totalnans} nans")
print(f"skipped {skippedtotal} columns for being boring")
    
    