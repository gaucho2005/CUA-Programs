#renames all the files in a folder
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
p= Path("/mnt/c/Users/ciroj/Desktop/Parquet Files/TnP/DY_postEE_2022/nominal/Columns")
folder = p
FolderName=p.parts[-3]
parquet_files = list(folder.glob("*.parquet"))
for f in parquet_files:
    newname=FolderName+"_"+f.stem
    print(newname)
    f.rename(f.parent / f"{newname}.parquet")