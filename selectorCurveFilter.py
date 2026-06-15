#A second selector program that applies a bin based mass fit filtering function, using a crystal ball function 
import pandas as pd
import vector
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import pyarrow.parquet as pq
import polars as pl
from Efficiency import calcEfficiency
import SpecialForces as SF
import subprocess
import plotly.graph_objects as go
import CurveFitEvilOG
import CurveFitEvilPointy

def choose_section(BorE):
    if BorE not in {"1", "2"}:
        BorE = input("Barrel (1) or Endcap (2)? ")
        while BorE not in {"1", "2"}:
            BorE = input("Barrel (1) or Endcap (2)? ")
    return "Barrel" if BorE == "1" else "Endcap"
    
def choose_axis(ETAorPT):
    if ETAorPT not in {"1", "2"}:
        ETAorPT = input("ETA (1) or PT (2)? ")

        while ETAorPT not in {"1", "2"}:
            ETAorPT = input("ETA (1) or PT (2)? ")
    return "probe_superclusterEta" if ETAorPT == "1" else "probe_pt"

def grouper(df1,BorE): #applies selections, then groups data into TT, TP, and TF
    #For the sake of sanity, filtering out the outlier high values of pt
    df = df1.filter(pl.col("probe_pt") < 225)
    
    
    
    #Cut to distinguish between the barrel and the endcap
    if BorE=="Barrel":
        df = df.filter((pl.col("probe_superclusterEta") < 1.4442) & (pl.col("probe_superclusterEta") >-1.4442)) #barrel
        sieieCriteria=.0104
    if BorE=="Endcap":
        #df = df.filter((2.5 > pl.col("probe_superclusterEta") > 1.56) | (-2.5 < pl.col("probe_superclusterEta") < -1.56)) #endcap
        df = df.filter((pl.col("probe_superclusterEta").is_between(1.56,2.5)) | (pl.col("probe_superclusterEta").is_between(-2.5,-1.56))) #endcap
        sieieCriteria=.0353
        
    #First filter out all rows with tag electrons that don't meet the test criteria
    olddf=df
    df=olddf.filter((olddf["tag_sieie"]<sieieCriteria) & (olddf["probe_mvaID_WP80"]==True))
    print(f"{(len(olddf)-len(df)):,} ROWS ELIMINATED FOR NOT HAVING VALID TAG (FAILED TO MEET TEST CRITERIA)")
    

    #Then find our TTs, which will be all those rows in which the probe electron meets the Tag specs, and then the test criteria
    TagTag= df.filter((df["probe_pt"]>40) & (df["probe_electronIdx"] != -1) & (df["probe_pfChargedIsoPFPV"] <20)& ((df["probe_pfChargedIsoPFPV"]/df["probe_pt"]) <.3)) #checks which probes would meet the tag criteria
    TT=TagTag.filter(TagTag["probe_sieie"]<sieieCriteria)
    print(f"{len(TT):,} ROWS HAVE PROBES THAT MEET BOTH TAG AND TEST CRITERIA (TT)")


    #Then find our TPs, which will be all the rows in which the probe electron does not meet the tag specs, but does meet the test criteria
    TagProbe= df.filter((df["probe_pt"]<=40) | (df["probe_electronIdx"] == -1) | (df["probe_pfChargedIsoPFPV"] >=20) | ((df["probe_pfChargedIsoPFPV"]/df["probe_pt"]) >=.3))
    TP=TagProbe.filter(TagProbe["probe_sieie"]<sieieCriteria)
    print(f"{len(TP):,} ROWS HAVE PROBES THAT FAIL TAG CRITERA BUT MEET TEST CRITERIA (TP)")

    #Then find our TFs, which will be all the rows in which the probe electron does not meet the tag specs, and does not meet the test criteria
    TagFail= df.filter((df["probe_pt"]<=40) | (df["probe_electronIdx"] == -1) | (df["probe_pfChargedIsoPFPV"] >=20) | ((df["probe_pfChargedIsoPFPV"]/df["probe_pt"]) >=.3))
    TF=TagFail.filter(TagFail["probe_sieie"]>=sieieCriteria)
    print(f"{len(TF):,} ROWS HAVE PROBES THAT ARE UTTER FAILURES (TF)")

    """
    half = len(TT) // 2
    TT = TT.head(half) #Do we need to multiply TT by 2 in the Efficiency equation? Must think about how the rows work in these data files, there is double counting.
    """
    return TT,TP,TF
def binning(df,axis):
    rows = []
    if axis=="probe_pt":
        for low in range(0, 120, 10):
            high = low + 10

            subset = df.filter(
                (pl.col(axis) >= low) &
                (pl.col(axis) < high)
            )
            if len(subset) == 0:
               value=0
            else:
                value = CurveFitEvilPointy.main(subset)
            
            rows.append({
                "low": low,
                "high": high,
                "center": (low + high) / 2,
                "Nvalue": value
        })
    if axis=="probe_superclusterEta":
        for low in np.linspace(-2.49, 2.49, 30):
            high = low + .166

            subset = df.filter(
                (pl.col(axis) >= low) &
                (pl.col(axis) < high)
            )
            if len(subset) == 0:
               value=0
            else:
                value = CurveFitEvilPointy.main(subset)
            
            rows.append({
                "low": low,
                "high": high,
                "center": (low + high) / 2,
                "Nvalue": value
        })
    return pl.DataFrame(rows)
    
def plotter(bin_centers, counts,error,DataName,section,axis):
    plt.close()
    # Plot Data
    plt.errorbar(bin_centers, counts, yerr=error,fmt='.',capsize=2,ecolor="black",color=SF.randColor(),elinewidth=1)
    plt.ticklabel_format(style='plain', axis='y')
    plt.xlabel(axis, fontsize=12)
    plt.ylabel('efficiency', fontsize=12)
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.ylim(.95, 1.05)
    title=f"{DataName} {section} Efficiency vs {axis} (Curve Signal)"
    plt.title(title, fontsize=14)
    out_folder = Path("/home/ciroj/CUA/testing/Charts")
    outfile = out_folder / f"{title}.png"
    print(f"GENERATING HISTOGRAM OF {title}")
    plt.savefig(outfile,bbox_inches='tight')
    subprocess.run(["explorer.exe", str(outfile).replace("/", "\\")],shell=False)
    
def makeHist(dataName,dataPath,section,axis):
    
    datas = (pl.scan_parquet(dataPath).select(["probe_superclusterEta","tag_superclusterEta","tag_pt","tag_sieie","tag_eta","probe_mvaID_WP80","probe_sieie","probe_pt","probe_eta","probe_electronIdx","probe_pfChargedIsoPFPV","mass"])).collect()
    TT,TP,TF=grouper(datas,section)
    TTdata=binning(TT,axis)
    TPdata=binning(TP,axis)
    TFdata=binning(TF,axis)
    numerator = TTdata.with_columns((pl.col("Nvalue") + TPdata["Nvalue"]).alias("Nvalue"))
    denominator=numerator.with_columns((pl.col("Nvalue") + TFdata["Nvalue"]).alias("Nvalue"))
    efficiencyvalues=numerator["Nvalue"]/denominator["Nvalue"]
    efficiencyError=np.sqrt((1-efficiencyvalues)*efficiencyvalues/(len(denominator["Nvalue"])))
    plotter(denominator["center"],efficiencyvalues,efficiencyError,dataName,section,axis)
    
    
def main():
    realData=Path(f"/mnt/c/Users/ciroj/Desktop/Parquet Files/TnP/MultifileHistograms/AllFilesData/AllRealData.parquet")
    simData=Path(f"/mnt/c/Users/ciroj/Desktop/Parquet Files/TnP/MultifileHistograms/AllFilesData/DY_postEE_2022.parquet")
    BorE="0"
    axis="0"
    section=choose_section(BorE)
    axis=choose_axis(axis)
    paths= {
        "Real Data":realData,
        "Sim Data":simData }
    for name,data in paths.items():
        makeHist(name,data,section,axis)
main()