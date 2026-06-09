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
use_cols = ["tag_energyRaw", "tag_pt", "tag_eta", "tag_phi", "probe_energyRaw", "probe_pt", "probe_eta", "probe_phi"]  # whatever you actually need

def Zmassfind(file):
    #file = "/mnt/c/Users/ciroj/Desktop/Parquet Files/TnP/DY_postEE_2022/nominal/"
    readfile= pd.read_parquet(file)
    tagEnergy= readfile["tag_energyRaw"]
    tagPt=readfile["tag_pt"]
    tagEta=readfile["tag_eta"]
    tagPhi=readfile["tag_phi"]
    probeEnergy= readfile["probe_energyRaw"]
    probePt=readfile["probe_pt"]
    probeEta=readfile["probe_eta"]
    probePhi=readfile["probe_phi"]

    tagPx=tagPt*(np.cos(tagPhi))
    tagPy=tagPt*(np.sin(tagPhi))
    tagPz=tagPt*(np.sinh(tagEta))
    probePx=probePt*(np.cos(probePhi))
    probePy=probePt*(np.sin(probePhi))
    probePz=probePt*(np.sinh(probeEta))

    ZPx=(tagPx+probePx)
    ZPy=(tagPy+probePy)
    ZPz=(tagPz+probePz)
    ZP=np.sqrt((np.square(ZPx))+(np.square(ZPy))+(np.square(ZPz)))
    ZEnergy=(tagEnergy+probeEnergy)
    valid= ((np.square(ZEnergy))-(np.square(ZP))) >=0
    ZM=np.sqrt(((np.square(ZEnergy))-(np.square(ZP)))[valid])

    ZM2=np.roll(ZM,-1)
    mask = ZM!=ZM2
    ZMfiltered=ZM[mask] #filters out sequentially repeated values
    return ZMfiltered

    

results = []

for f in tqdm(parquet_files, desc="Doing math on files"):
    zm = Zmassfind(f)   # returns an array / Series
    results.append(zm)

finalZM = np.concatenate(results)

plt.hist(finalZM, bins=60, range=(0,160),color='red')
plt.ticklabel_format(style='plain', axis='y')
plt.axvline(x=91.2, color='blue', linestyle='-', linewidth=2)
plt.savefig("hist.png")