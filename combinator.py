#combine all the parquet files into one big file
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
import pyarrow as pa
import pyarrow.parquet as pq

def combinate(FolderName):
    folder = Path(f"/mnt/c/Users/ciroj/Desktop/Parquet Files/TnP/MultifileHistograms/AllFilesData/AllRealData")
    out_folder = Path(f"/mnt/c/Users/ciroj/Desktop/Parquet Files/TnP/MultifileHistograms/AllFilesData")
    parquet_files = list(folder.glob("*.parquet"))


    tables = []

    for f in tqdm(parquet_files, desc="reading"):
        tables.append(pq.read_table(f))
    
    combined = pa.concat_tables(tables)
    pq.write_table(combined, out_folder / f"{FolderName}.parquet")
    
    
#filelist=["DataE_2022","DY_postEE_2022","DataF_2022","DataG_2022"]
filelist=["AllRealData"]
for i in tqdm(filelist,desc="Making single data files for each collection"):
    combinate(i)