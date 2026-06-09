#Applies Selectivity Criteria, Then Checks Efficiency
import pandas as pd
import vector
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
from pathlib import Path
from tqdm import tqdm
import pyarrow.parquet as pq
import polars as pl
from Efficiency import calcEfficiency
import SpecialForces as SF
realDataPath=Path(f"/mnt/c/Users/ciroj/Desktop/Parquet Files/TnP/MultifileHistograms/AllFilesData/AllRealData.parquet")
simDataPath=Path(f"/mnt/c/Users/ciroj/Desktop/Parquet Files/TnP/MultifileHistograms/AllFilesData/DY_postEE_2022.parquet")
def operate(datafilepath):
    #df1 = pl.read_parquet(datafilepath) #grabs the entire file as a table in polars
    df1 = (pl.scan_parquet(datafilepath).select(["tag_pt","tag_sieie","probe_mvaID_WP80","probe_sieie","probe_pt","probe_electronIdx","probe_pfChargedIsoPFPV",])).collect()

    #First filter out all rows with tag electrons that don't meet the test criteria
    df=df1.filter((df1["tag_sieie"]<.0104) & (df1["probe_mvaID_WP80"]==True))
    print(f"{(len(df1)-len(df)):,} ROWS ELIMINATED FOR NOT HAVING VALID TAG (FAILED TO MEET TEST CRITERIA)")

    #Then find our TTs, which will be all those rows in which the probe electron meets the Tag specs, and then the test criteria
    TagTag= df.filter((df["probe_pt"]>40) & (df["probe_electronIdx"] != -1) & (df["probe_pfChargedIsoPFPV"] <20)& ((df["probe_pfChargedIsoPFPV"]/df["probe_pt"]) <.3)) #checks which probes would meet the tag criteria
    TT=TagTag.filter(TagTag["probe_sieie"]<.0104)
    print(f"{len(TT):,} ROWS HAVE PROBES THAT MEET BOTH TAG AND TEST CRITERIA (TT)")


    #Then find our TPs, which will be all the rows in which the probe electron does not meet the tag specs, but does meet the test criteria
    TagProbe= df.filter((df["probe_pt"]<=40) | (df["probe_electronIdx"] == -1) | (df["probe_pfChargedIsoPFPV"] >=20) | ((df["probe_pfChargedIsoPFPV"]/df["probe_pt"]) >=.3))
    TP=TagProbe.filter(TagProbe["probe_sieie"]<.0104)
    print(f"{len(TP):,} ROWS HAVE PROBES THAT FAIL TAG CRITERA BUT MEET TEST CRITERIA (TP)")

    #Then find our TFs, which will be all the rows in which the probe electron does not meet the tag specs, and does not meet the test criteria
    TagFail= df.filter((df["probe_pt"]<=40) | (df["probe_electronIdx"] == -1) | (df["probe_pfChargedIsoPFPV"] >=20) | ((df["probe_pfChargedIsoPFPV"]/df["probe_pt"]) >=.3))
    TF=TagFail.filter(TagFail["probe_sieie"]>=.0104)
    print(f"{len(TF):,} ROWS HAVE PROBES THAT ARE UTTER FAILURES (TF)")

    """
    half = len(TT) // 2
    TT = TT.head(half) #Do we need to multiply TT by 2 in the efficiency equation? Must think about how the rows work in these data files, there is double counting.
    """
    epsilon=calcEfficiency(TT,TP,TF)
    percent=epsilon*100
    print(f"{percent:.3f}% SELECTION EFFICIENCY")
    print(f"{100-percent:.3f}% OF OUR ELECTRONS FAILED THE TEST")
    return epsilon
print("Operating on real data:")
Edata=operate(realDataPath)
print("Operating on simulation data:")
Esim=operate(simDataPath)
print(f"Scale ratio (data efficiency over sim efficiency) is {(Edata/Esim):.4f}")



