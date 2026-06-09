#break all the columns into separate files
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


out_folder = Path("/mnt/c/Users/ciroj/Desktop/Parquet Files/TnP/DataE_2022/nominal/Columns")
folder = Path("/mnt/c/Users/ciroj/Desktop/Parquet Files/TnP/DataE_2022/nominal/")
parquet_files = list(folder.glob("*.parquet"))
totalcolumn=[]
df=pd.read_parquet(parquet_files[1])
column_names = df.columns.tolist()

data_dict = {name: [] for name in column_names}
for f in tqdm(parquet_files,desc="combining columns"):
    df=pd.read_parquet(f)
    for name in column_names:
        #data_dict[name].extend(df[name].to_numpy())
        data_dict[name].append(df[name].to_numpy())
    
for name in tqdm(column_names,desc= "concatenating"):
    data_dict[name] = np.concatenate(data_dict[name])
    
for name, values in tqdm(data_dict.items(),desc= "making files"):
    df = pd.DataFrame({name: values})
    df.to_parquet(out_folder / f"{name}.parquet")