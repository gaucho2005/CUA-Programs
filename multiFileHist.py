#should probably try making all my steps functions then just repeating the functions for each file? 
#make histograms out of a file full of column files, applies some filtering
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
    if np.unique(col).size <= 1000:
        return True

    return False

filesfolder = Path(f"/mnt/c/Users/ciroj/Desktop/Parquet Files/TnP/MultifileHistograms/AllFilesData")
folderfiles = list(filesfolder.iterdir())
out_folder = Path("/mnt/c/Users/ciroj/Desktop/Parquet Files/TnP/MultifileHistograms")

 

totalnans=0
def genHist(col): #at present moment, generates a histogram for each column given
    nan_count = np.isnan(col).sum()
    #print(f"Found and removed {nan_count} NaN values.")
    total=totalnans+nan_count
    col = col[np.isfinite(col)] #finds and removes nans
    
    counts, bin_edges = np.histogram(col, bins=100)
    
    if counts.max()>1100000: #implements different rules for histograms that would have an excessive amount of data concentrated in one bin
        counts, bin_edges = np.histogram(col, bins=350) #was 300
        
        indices = np.argsort(-counts) #if the highest bar is more than double the second highest, cut it off
        sorted_values = counts[indices]
        if sorted_values[0]>(2*sorted_values[1]): 
            plt.ylim(top=(1.1*sorted_values[1]),bottom=0)
            mask= counts>=(.005*sorted_values[1])
            filtered_counts = counts[mask]
        else:
            mask= counts>=(.005*sorted_values[0])
            filtered_counts = counts[mask]
           
        

    else:
        mask = counts >= 1000
        filtered_counts = counts[mask] #filters out bars with less than a certain number of entries
    
    print(f"Max bar is {filtered_counts.max()}")
    filtered_edges = bin_edges[:-1][mask]
    width = np.diff(bin_edges)[0]
    plt.bar(filtered_edges, filtered_counts, width=width, color='red')
    plt.ticklabel_format(style='plain', axis='y')
def function1(parquet_files): #Selects what is to be histogrammed, then sends it to be made
    for f in tqdm(parquet_files,desc="makinghistograms"):
        #df=pd.read_parquet(f)
        #columnname = df.iloc[:,0]
        columnname = pq.read_table(f).column(0).to_numpy()
        print(columname)
        """
        pf = pq.ParquetFile(f)
        title=(pf.schema.names[0]) #this plus line before makes for much faster reading of names, maybe use that for something later
        
        if title !="tag_trkSumPtSolidConeDR04":
            continue
        
        df=pd.read_parquet(f)
        col = df.iloc[:,0]
        sorted_col= col.sort_values(ascending=False)
        print (sorted_col)
        print(col.nlargest(10).values) #we'll use this is we need speed    
        """
        name= f.stem
        print(name)
        if skip_column(columnname):
            print("skipping invalid column (contains booleans or too few unique values)")
            global skippedtotal
            skippedtotal+=1
            continue
        
        """
        histmax=col.max()
        histmin=col.min()
        print("max=",histmax)
        print("min=",histmin)
        """
        genHist(columnname)

        filename = f"{f.stem}_hist.png"
        
        
        #plt.savefig(out_folder / filename)
        plt.close()
def function2(parquet_files): #Must needs be rewritten for handling multiple big parquet files instead of hundreds of small ones
    #first, we must generate a column list so we can iterate by columns
    pf = pq.ParquetFile(f)
        title=(pf.schema.names[0])
    
    
    
    
    for f in tqdm(parquet_files,desc="makinghistograms"):
        #df=pd.read_parquet(f)
        #columnname = df.iloc[:,0]
        columnname = pq.read_table(f).column(0).to_numpy()
        print(columname)
        """
        pf = pq.ParquetFile(f)
        title=(pf.schema.names[0]) #this plus line before makes for much faster reading of names, maybe use that for something later
        
        if title !="tag_trkSumPtSolidConeDR04":
            continue
        
        df=pd.read_parquet(f)
        col = df.iloc[:,0]
        sorted_col= col.sort_values(ascending=False)
        print (sorted_col)
        print(col.nlargest(10).values) #we'll use this is we need speed    
        """
        name= f.stem
        print(name)
        if skip_column(columnname):
            print("skipping invalid column (contains booleans or too few unique values)")
            global skippedtotal
            skippedtotal+=1
            continue
        
        """
        histmax=col.max()
        histmin=col.min()
        print("max=",histmax)
        print("min=",histmin)
        """
        genHist(columnname)

        filename = f"{f.stem}_hist.png"
        
        
        #plt.savefig(out_folder / filename)
        plt.close()
def main():
    #folder = Path(f"/mnt/c/Users/ciroj/Desktop/Parquet Files/TnP/{k}/nominal/Columns")
    #parquet_filelist = list(k.glob("*.parquet"))
    
    totalnans=0
    function2(folderfiles)
    print(f"found total {totalnans} nans")
    print(f"skipped {skippedtotal} columns for being boring") 
main()        
    