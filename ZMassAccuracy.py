import pandas as pd
import vector
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import math
pd.set_option("display.max_rows", 75)
from pathlib import Path
from tqdm import tqdm
import vector
import awkward as ak
import subprocess

folder = Path("/mnt/c/Users/ciroj/Desktop/Parquet Files/TnP/DY_postEE_2022/nominal/")
parquet_files = list(folder.glob("*.parquet"))
use_cols = ["tag_energyRaw", "tag_pt", "tag_eta", "tag_phi", "probe_energyRaw", "probe_pt", "probe_eta", "probe_phi"]  # whatever you actually need

def Zmassfind(file):
    df = pd.read_parquet(file)
    realmass=df["mass"]
    e_mass = 0.000511
    # 1. Create 4-vectors using the 'zip' method
    # It is best practice to use .to_numpy() for compatibility
    tag_vec = vector.zip({
        "pt": df["tag_pt"].to_numpy(),
        "eta": df["tag_eta"].to_numpy(),
        "phi": df["tag_phi"].to_numpy(),
        "mass": np.full(len(df), e_mass) # Assign the electron mass to every event
    })

    probe_vec = vector.zip({
        "pt": df["probe_pt"].to_numpy(),
        "eta": df["probe_eta"].to_numpy(),
        "phi": df["probe_phi"].to_numpy(),
        "mass": np.full(len(df), e_mass) 
    })

    
    z_boson=probe_vec+tag_vec
    ZM = z_boson.mass

    ZMdif = ZM-realmass
    ZMdifPcent = (ZMdif / realmass) * 100 #calculates the percent deviation from the real mass for each mass
    return ZM, ZMdifPcent

    

results = []
difPcentArray=[]

for f in tqdm(parquet_files, desc="Doing math on files"):
    #print(f"filenameis {f}")
    zm, difPcentVals = Zmassfind(f)  # Unpack the two return values
    results.append(zm)
    difPcentArray.append(difPcentVals)
    

finalreal = np.concatenate(results)
finalDifPcentArray=np.concatenate(difPcentArray)
positive=finalDifPcentArray>0
positivefinalDifPcentArray=finalDifPcentArray[positive]
print(f"The average positive percent deviation from the real mass is {np.nanmean(positivefinalDifPcentArray)}")
print(f"The average percent deviation from the real mass is {np.nanmean(finalDifPcentArray)}")
plt.hist(finalDifPcentArray, bins=100, range=(-90,12),color='red')
plt.ticklabel_format(style='plain', axis='y')
plt.axvline(x=0, color='blue', linestyle='-', linewidth=2)
plt.savefig("pcentDeviationHist.png")
subprocess.run("explorer.exe pcentDeviationHist.png", shell=True)

