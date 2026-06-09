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
    
    """
    ZM2 = ak.concatenate([ZM[1:], ZM[:1]]) #remove sequential repetition    
    mask = ~np.isclose(ZM, ZM2, atol=1e-5)
    """
    
    # This identifies indices where the value is different from the previous one
    diff_mask = np.concatenate(([True], ~np.isclose(ZM[1:], ZM[:-1], atol=1e-5)))
    ZMfiltered = ZM[diff_mask]
    Zremoved=len(ZM)-np.sum(mask)
    
    #valid=ZM>0
    #print(f"{len(ZM)-len(ZM[valid])}")
    #ZM=ZM[valid] #removes the values of ZM that are less than zero for some reason NO LONGER NECESSARY
    
    return ZMfiltered, Zremoved, realmass

    

results = []
Zremovedtotal= 0
realmassbig = []
for f in tqdm(parquet_files, desc="Doing math on files"):
    zm, z_removed_count,realmassvalues = Zmassfind(f)  # Unpack the two return values
    results.append(zm)
    realmassbig.append(realmassvalues)
    Zremovedtotal=Zremovedtotal+z_removed_count

finalZM = np.concatenate(results)
concatenatedrealmass=np.concatenate(realmassbig)
print(f"Removed {Zremovedtotal} sequentially repeated columns")
plt.hist(finalZM, bins=150, range=(0,160),color='green',alpha=0.5, label="calculated mass")
plt.hist(concatenatedrealmass, bins=150, range=(0,160),color='blue',alpha=0.5, label="recorded mass")
plt.ticklabel_format(style='plain', axis='y')
plt.axvline(x=91.2, color='red', linestyle='-', linewidth=2)
plt.legend()
plt.savefig("hist2.png")
subprocess.run("explorer.exe hist2.png", shell=True)
plt.savefig("massComparisonHist.png")
subprocess.run("explorer.exe massComparisonHist.png", shell=True)