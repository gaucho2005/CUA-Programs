#check if all parquet files have the same columns
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


folder = Path("/mnt/c/Users/ciroj/Desktop/Parquet Files/TnP/DY_postEE_2022/nominal/")
parquet_files = list(folder.glob("*.parquet"))
totalcolumn=[]
df1=pd.read_parquet(parquet_files[0])
list2=df1.columns.tolist()
for i in tqdm(range(0,len(parquet_files)),desc="running"):
    df=pd.read_parquet(parquet_files[i])
    list1 = df.columns.tolist()
    if (list1==list2):
        pass
    else:
        print("failure to match columns") 
    list2=df.columns.tolist()