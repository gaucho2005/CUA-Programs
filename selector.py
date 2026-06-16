#Applies Selectivity Criteria, Can Then Check Efficiency, or Can Plot 2D or 3D Histogram
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
realData=Path(f"/mnt/c/Users/ciroj/Desktop/Parquet Files/TnP/MultifileHistograms/AllFilesData/AllRealData.parquet")
simData=Path(f"/mnt/c/Users/ciroj/Desktop/Parquet Files/TnP/MultifileHistograms/AllFilesData/DY_postEE_2022.parquet")
    
    
def grouper(df1,BorE): #applies selections, then groups data into TT, TP, and TF
    #For the sake of sanity, filtering out the outlier high values of pt
    df = df1.filter(pl.col("probe_pt") < 225)
    
    #Basic mass cut
    df=df.filter(pl.col("mass").is_between(80,110))
    
    
    #Cut to distinguish between the barrel and the endcap
    if BorE=="Barrel":
        df = df.filter((pl.col("probe_superclusterEta") < 1.4442) & (pl.col("probe_superclusterEta") >-1.4442)) #barrel
        sieieCriteria=.0104
    if BorE=="Endcap":
        #df = df.filter((2.5 > pl.col("probe_superclusterEta") > 1.56) | (-2.5 < pl.col("probe_superclusterEta") < -1.56)) #endcap
        df = df.filter((pl.col("probe_superclusterEta").is_between(1.56,2.5)) | (pl.col("probe_superclusterEta").is_between(-2.5,-1.56))) #endcap
        sieieCriteria=.0353
        print(len(df))
        
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
    
def getEpsilon(TT,TP,TF): #not really used in this program since we're doing epsilon=Npass/(Npass+Nfail)
    epsilon=calcEfficiency(TT,TP,TF)
    percent=epsilon*100
    print(f"{percent:.3f}% SELECTION EFFICIENCY")
    print(f"{100-percent:.3f}% OF OUR ELECTRONS FAILED THE TEST")
    return epsilon
    


def histdata(category,TT,TP,TF,binnumber):
    numerator=pl.concat([TT,TP])
    denominator=pl.concat([TT,TP,TF])
    numeratorcount, bins= np.histogram(numerator[category],bins=binnumber)
    denominatorcount,_ = np.histogram(denominator[category],bins=bins)
    
    efficiencycount=(numeratorcount/denominatorcount)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    # sqrt[(1-efficiency)*efficiency/Ntotal]?
    efficiencyError=np.sqrt((1-efficiencycount)*efficiencycount/(len(denominatorcount)))
    return efficiencycount,bin_centers,bins,efficiencyError

""" #some debugging functions
print(f"THE MAXIMUM IS {TT['probe_eta'].max()}")
    top10 = (
        TT["probe_eta"]
        .sort(descending=True)
        .head(50)
    )
    print(top10)
    
    plt.hist(denominator['probe_eta'], bins=500, range=(-3,3),color=SF.randColor(),label="probe_eta")
    has_zeros = (numeratorcount == 0).any()
    print(f"DOES IT HAVE ZEROS???????? {has_zeros}")
"""
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

def makeHist(DataName,DataPath,section,axis,bins): #makes 2d histograms with one bin axis
    plt.close() #refreshes plt so there's no overlapping charts for looped hist creation
    datas = (pl.scan_parquet(DataPath).select(["probe_superclusterEta","tag_superclusterEta","tag_pt","tag_sieie","tag_eta","probe_mvaID_WP80","probe_sieie","probe_pt","probe_eta","probe_electronIdx","probe_pfChargedIsoPFPV","mass"])).collect()
    print(f"Generating {DataName} Hist")
    TT,TP,TF=grouper(datas,section)
    PTefficiencycount,PTbin_centers,PTbins,PTerror=histdata("probe_pt",TT,TP,TF,bins)
    ETAefficiencycount,ETAbin_centers,ETAbins,ETAerror=histdata("probe_superclusterEta",TT,TP,TF,bins)
    if axis=="probe_superclusterEta":
        plt.errorbar(ETAbin_centers,ETAefficiencycount,yerr=ETAerror,fmt='.',capsize=2,ecolor="black",color=SF.randColor(),elinewidth=1)
    if axis=="probe_pt":
        plt.errorbar(PTbin_centers,PTefficiencycount,yerr=PTerror,fmt='.',capsize=2,ecolor="black",color=SF.randColor(),elinewidth=1)
    plt.ylim(.95, 1.05)
    title=f"{DataName} {section} Efficiency vs {axis}"
    out_folder = Path("/home/ciroj/CUA/testing/Charts")
    outfile = out_folder / f"{title}.png"
    plt.savefig(outfile)
    subprocess.run(["explorer.exe", str(outfile).replace("/", "\\")],shell=False)
    
def gen3dhistdata(TT,TP,TF): #generates info necessary to plot a 3d histogram  
    numerator=pl.concat([TT,TP])
    denominator=pl.concat([TT,TP,TF])
    x_num = numerator["probe_pt"].to_numpy()
    y_num = numerator["probe_superclusterEta"].to_numpy()
    x_den = denominator["probe_pt"].to_numpy()
    y_den = denominator["probe_superclusterEta"].to_numpy()
    xedges = np.linspace(-25, 275, 28)
    yedges = np.linspace(-3, 3, 40)

    H_num, xedges, yedges = np.histogram2d(x_num, y_num, bins=(xedges, yedges))
    H_den, _, _          = np.histogram2d(x_den, y_den, bins=(xedges, yedges))

    efficiency = np.divide(H_num,H_den,out=np.zeros_like(H_num, dtype=float),where=H_den != 0)
    
    # 3. Calculate bin centers
    x_centers = 0.5 * (xedges[:-1] + xedges[1:])
    y_centers = 0.5 * (yedges[:-1] + yedges[1:])
    efficiency = efficiency.T
    return x_centers,y_centers,efficiency

def make3DHist(): #makes 3d histograms with 2 bin axes
    real_data = (pl.scan_parquet(realDataPath).select(["probe_superclusterEta","tag_superclusterEta","tag_pt","tag_sieie","tag_eta","probe_mvaID_WP80","probe_sieie","probe_pt","probe_eta","probe_electronIdx","probe_pfChargedIsoPFPV",])).collect()
    sim_data = (pl.scan_parquet(simDataPath).select(["probe_superclusterEta","tag_superclusterEta","tag_pt","tag_sieie","tag_eta","probe_mvaID_WP80","probe_sieie","probe_pt","probe_eta","probe_electronIdx","probe_pfChargedIsoPFPV",])).collect()
    TT,TP,TF=grouper(real_data)
    real_x_centers,real_y_centers,real_efficiency=gen3dhistdata(TT,TP,TF)
    TT,TP,TF=grouper(sim_data)
    sim_x_centers,sim_y_centers,sim_efficiency=gen3dhistdata(TT,TP,TF)
    
    #Factor=real_efficiency/sim_efficiency

    # 4. Create the surface plot 
    fig = go.Figure(data=[go.Surface(z=Factor, x=sim_x_centers, y=sim_y_centers,showscale=True,contours={"x": {"show": True, "size": 1},"y": {"show": True, "size": 1},"z": {"show": True}})])
    #fig = go.Figure(data=[go.Surface(z=sim_efficiency, x=sim_x_centers, y=sim_y_centers,showscale=True,contours={"x": {"show": True, "size": 1},"y": {"show": True, "size": 1},"z": {"show": True}})])
    #fig = go.Figure(data=[go.Surface(z=efficiency, x=x_centers, y=y_centers)])
    
    # 5. Format the layout for 3D visibility
    fig.update_layout(
        title="Scale Factor Endcap Data Photon-Faking-Electron Efficiency 3D Histogram",
        autosize=False,
        width=800,
        height=800,
        scene=dict(
            xaxis_title='probe_pt',
            yaxis_title='probe_superclusterEta',
            zaxis_title='Efficiency'
        )
    )
    fig.update_layout(
    scene=dict(xaxis=dict(range=[-25, 275]),yaxis=dict(range=[-3, 3]),zaxis=dict(range=[.985, 1.015])))
    fig.show()
    title="ScaleFactorEndcap3DHist"
    out_folder = Path("/home/ciroj/CUA/testing/Charts")
    outfile = out_folder / f"{title}.html"
    fig.write_html(outfile)
    
    subprocess.run(["explorer.exe", str(outfile).replace("/", "\\")],shell=False)

def main():
    BorE="0"
    axis="0"
    section=choose_section(BorE)
    axis=choose_axis(axis)
    paths= {
        "Real Data":realData,
        "Sim Data":simData }
    for name,data in paths.items():
        makeHist(name,data,section,axis,100)
main()   




