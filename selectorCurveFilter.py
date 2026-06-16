#A second selector program that applies a bin based mass fit filtering function, using a crystal ball function, to make 2D hists
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
import readchar
import itertools

def choose_section(BorE):
    if BorE not in {"1", "2"}:
        while True:
            print("Barrel (1) or Endcap (2)? ", end="", flush=True)
            BorE = readchar.readkey()
            print(BorE)

            if BorE in {"1", "2"}:
                break

    return "Barrel" if BorE == "1" else "Endcap"
    
def choose_axis(ETAorPT):
    if ETAorPT not in {"1", "2"}:
        while True:
            print("ETA (1) or PT (2)? ", end="", flush=True)
            ETAorPT = readchar.readkey()
            print(ETAorPT)

            if ETAorPT in {"1", "2"}:
                break
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
    #df=olddf.filter((olddf["tag_sieie"]<sieieCriteria) & (olddf["probe_mvaID_WP80"]==True))
    #df=olddf.filter(olddf["tag_sieie"]<sieieCriteria)
    #df=olddf.filter(olddf["tag_mvaID_WP80"]==True)
    print(f"{(len(olddf)-len(df)):,} ROWS ELIMINATED FOR NOT HAVING VALID TAG (FAILED TO MEET TEST CRITERIA)")
    

    #Then find our TTs, which will be all those rows in which the probe electron meets the Tag specs, and then the test criteria
    TagTag= df.filter((df["probe_pt"]>40) & (df["probe_electronIdx"] != -1) & (df["probe_pfChargedIsoPFPV"] <20)& ((df["probe_pfChargedIsoPFPV"]/df["probe_pt"]) <.3)) #checks which probes would meet the tag criteria
    #TT=TagTag.filter((TagTag["probe_sieie"]<sieieCriteria) & (TagTag["probe_mvaID_WP80"]==True))
    #TT=TagTag.filter(TagTag["probe_mvaID_WP80"]==True)
    TT=TagTag
    TT=TT.filter(TT["probe_pixelSeed"]==False)
    #TT=TagTag.filter(TagTag["probe_sieie"]<sieieCriteria)
    print(f"{len(TT):,} ROWS HAVE PROBES THAT MEET BOTH TAG AND TEST CRITERIA (TT)")


    #Then find our TPs, which will be all the rows in which the probe electron does not meet the tag specs, but does meet the test criteria
    TagProbe= df.filter((df["probe_pt"]<=40) | (df["probe_electronIdx"] == -1) | (df["probe_pfChargedIsoPFPV"] >=20) | ((df["probe_pfChargedIsoPFPV"]/df["probe_pt"]) >=.3))
    #TP=TagProbe.filter((TagProbe["probe_sieie"]<sieieCriteria) & (TagProbe["probe_mvaID_WP80"]==True))
    #TP=TagProbe.filter(TagProbe["probe_sieie"]<sieieCriteria)
    #TP=TagProbe.filter(TagProbe["probe_mvaID_WP80"]==True)
    TP=TagProbe
    TP=TP.filter(TP["probe_pixelSeed"]==False)
    print(f"{len(TP):,} ROWS HAVE PROBES THAT FAIL TAG CRITERA BUT MEET TEST CRITERIA (TP)")

    #Then find our TFs, which will be all the rows in which the probe electron does not meet the tag specs, and does not meet the test criteria
    #TagFail= df.filter((df["probe_pt"]<=40) | (df["probe_electronIdx"] == -1) | (df["probe_pfChargedIsoPFPV"] >=20) | ((df["probe_pfChargedIsoPFPV"]/df["probe_pt"]) >=.3))
    #TF=TagFail.filter((TagFail["probe_sieie"]>=sieieCriteria) | (TagFail["probe_mvaID_WP80"]!=True))
    #TF=TagFail.filter(TagFail["probe_sieie"]>=sieieCriteria)
    #TF=TagFail.filter((TagFail["probe_mvaID_WP80"]!=True) | (TagFail["probe_pixelSeed"]==True))
    TF=df.filter(df["probe_pixelSeed"]==True)
    #TF=TF.filter(TF["probe_mvaID_WP80"]==True)
    print(f"{len(TF):,} ROWS HAVE PROBES THAT ARE UTTER FAILURES (TF)")

    """
    half = len(TT) // 2
    TT = TT.head(half) #Do we need to multiply TT by 2 in the Efficiency equation? Must think about how the rows work in these data files, there is double counting.
    """
    return TT,TP,TF
def binning(df,axis):
    rows = []
    if axis=="probe_pt":
        for low in tqdm(range(0, 120, 10), desc="Let me Cook"):
            high = low + 10

            subset = df.filter(
                (pl.col(axis) >= low) &
                (pl.col(axis) < high)
            )
            
            if len(subset) == 0:
               value=0
            else:
                print(f"EEEEEEEEEEEEEEEEEEEEEE THE POINT IN QUESTION IS {(high+low)/2}")
                value = CurveFitEvilPointy.main(subset) #sends that bin to another program to apply a Zmass fit
            
            rows.append({
                "low": low,
                "high": high,
                "center": (low + high) / 2,
                "Nvalue": value
        })
    if axis=="probe_superclusterEta":
        for low in tqdm(np.linspace(-2.49, 2.49, 30),desc="Let me Cook"):
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
    plt.ylim(0,.1)
    title=f"{DataName} {section} Electron Faking Efficiency vs {axis}"
    #title=f"{section} Efficiency vs {axis} Scale Factor"
    plt.title(title, fontsize=14)
    out_folder = Path("/home/ciroj/CUA/testing/Charts/Electron Faking With Vs Without MVAID80")
    outfile = out_folder / f"{title}.png"
    print(f"GENERATING HISTOGRAM OF {title}")
    plt.savefig(outfile,bbox_inches='tight')
    subprocess.run(["explorer.exe", str(outfile).replace("/", "\\")],shell=False)
    
def makeHist(dataName,dataPath,section,axis):
    
    datas = (pl.scan_parquet(dataPath).select(["probe_superclusterEta","tag_superclusterEta","tag_pt","tag_sieie","tag_eta","tag_mvaID_WP80","probe_mvaID_WP80","probe_sieie","probe_pt","probe_eta","probe_electronIdx","probe_pfChargedIsoPFPV","tag_pixelSeed","probe_pixelSeed","mass"])).collect()
    TT,TP,TF=grouper(datas,section)
    print("Binning TT!") ;  TTdata=binning(TT,axis)
    print("Binning TP!"); TPdata=binning(TP,axis)
    print("Binning TF!"); TFdata=binning(TF,axis)
    numerator = TTdata.with_columns((pl.col("Nvalue") + TPdata["Nvalue"]).alias("Nvalue"))
    denominator=numerator.with_columns((pl.col("Nvalue") + TFdata["Nvalue"]).alias("Nvalue"))
    efficiencyvalues=numerator["Nvalue"]/denominator["Nvalue"]
    efficiencyError=np.sqrt((1-efficiencyvalues)*efficiencyvalues/(len(denominator["Nvalue"])))
    plotter(denominator["center"],efficiencyvalues,efficiencyError,dataName,section,axis)
    """
    if dataName=="Real Data": #use this when you want to get the scale factor
        global efficiencyvalues
        efficiencyvalues=numerator["Nvalue"]/denominator["Nvalue"]
    if dataName=="Sim Data":
        efficiencyvalues2=numerator["Nvalue"]/denominator["Nvalue"]
        efficiencyvalues=efficiencyvalues/efficiencyvalues2
        efficiencyError=np.sqrt((1-efficiencyvalues)*efficiencyvalues/(len(denominator["Nvalue"])))
        plotter(denominator["center"],efficiencyvalues,efficiencyError,dataName,section,axis)
      
    """
    

def binning_3d(df):
    """
    Slices the dataframe across a 2D grid of pt and eta, runs the Z-mass fit 
    on every single intersecting bin, and returns grid data for a 3D plot.
    """
    # 1. Define the bin boundaries exactly like your original loops
    pt_lows = np.arange(0, 120, 10)
    eta_lows = np.linspace(-2.49, 2.49, 30)
    
    # 2. Pre-calculate bin centers for the output
    x_centers = pt_lows + 5.0                  # (low + (low + 10)) / 2
    y_centers = eta_lows + (0.166 / 2)         # (low + (low + 0.166)) / 2
    
    # 3. Initialize an empty 2D grid to hold the fitted Z-mass values
    # Shape is (len(pt), len(eta)) to map directly to our loops
    values_grid = np.zeros((len(pt_lows), len(eta_lows)))
    
    # Total iterations for the progress bar (e.g., 12 * 30 = 360 fits)
    total_bins = len(pt_lows) * len(eta_lows)
    
    # 4. Use product to loop through every X/Y combination simultaneously
    grid_iterator = itertools.product(enumerate(pt_lows), enumerate(eta_lows))
    
    for (i, pt_low), (j, eta_low) in tqdm(grid_iterator, total=total_bins, desc="Let me Cook"):
        pt_high = pt_low + 10
        eta_high = eta_low + 0.166
        
        # Filter the dataframe for events falling into this specific 2D box
        subset = df.filter(
            (pl.col("probe_pt") >= pt_low) & 
            (pl.col("probe_pt") < pt_high) &
            (pl.col("probe_superclusterEta") >= eta_low) & 
            (pl.col("probe_superclusterEta") < eta_high)
        )
        
        # If the bin is empty, assign 0; otherwise, run your Z-mass fitter
        if len(subset) == 0:
            value = 0.0
        else:
            value = CurveFitEvilPointy.main(subset)
            
        # Store the result in the corresponding grid coordinate
        values_grid[i, j] = value

    # 5. Transpose the matrix at the end to match your gen3dhistdata layout
    return x_centers, y_centers, values_grid.T
    
def make3DHist(dataname,datapath,section): #makes 3d histograms with 2 bin axes
    """
    real_data = (pl.scan_parquet(datapath).select(["probe_superclusterEta","tag_superclusterEta","tag_pt","tag_sieie","tag_eta","probe_mvaID_WP80","probe_sieie","probe_pt","probe_eta","probe_electronIdx","probe_pfChargedIsoPFPV",])).collect()
    sim_data = (pl.scan_parquet(simDataPath).select(["probe_superclusterEta","tag_superclusterEta","tag_pt","tag_sieie","tag_eta","probe_mvaID_WP80","probe_sieie","probe_pt","probe_eta","probe_electronIdx","probe_pfChargedIsoPFPV",])).collect()
    TT,TP,TF=grouper(real_data,section)
    x_centers,y_centers,realTTdata=binning_3d(TT) ; _,_,realTPdata=binning_3d(TP) ; _,_,realTFdata=binning_3d(TF)
    TT,TP,TF=grouper(sim_data)
    _,_,simTTdata=binning_3d(TT) ; _,_,simTPdata=binning_3d(TP) ; _,_,simTFdata=binning_3d(TF)
    """
    
    chartdata = (pl.scan_parquet(datapath).select(["probe_superclusterEta","tag_superclusterEta","tag_pt","tag_sieie","tag_eta","tag_mvaID_WP80","probe_mvaID_WP80","probe_sieie","probe_pt","probe_eta","probe_electronIdx","probe_pfChargedIsoPFPV","tag_pixelSeed","probe_pixelSeed","mass"])).collect()
    TT,TP,TF=grouper(chartdata,section)
    x_centers,y_centers,TTdata=binning_3d(TT) ; _,_,TPdata=binning_3d(TP) ; _,_,TFdata=binning_3d(TF)
    #Factor=real_efficiency/sim_efficiency
    numerator = TTdata+TPdata
    denominator=numerator+TFdata
    efficiencyvalues=numerator/denominator
    # 4. Create the surface plot 
    #fig = go.Figure(data=[go.Surface(z=Factor, x=sim_x_centers, y=sim_y_centers,showscale=True,contours={"x": {"show": True, "size": 1},"y": {"show": True, "size": 1},"z": {"show": True}})])
    #fig = go.Figure(data=[go.Surface(z=sim_efficiency, x=sim_x_centers, y=sim_y_centers,showscale=True,contours={"x": {"show": True, "size": 1},"y": {"show": True, "size": 1},"z": {"show": True}})])
    fig = go.Figure(data=[go.Surface(z=efficiencyvalues, x=x_centers, y=y_centers)])
    
    charttitle=f"{dataname} {section} Electron Faking Rate 3D Hist"
    fig.update_layout(
        title=charttitle,
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
    scene=dict(xaxis=dict(range=[-25, 200]),yaxis=dict(range=[-3, 3]),zaxis=dict(range=[-.5,.25])))
    fig.show()
    
    out_folder = Path("/home/ciroj/CUA/testing/Charts")
    outfile = out_folder / f"{charttitle}.html"
    fig.write_html(outfile)
    
    subprocess.run(["explorer.exe", str(outfile).replace("/", "\\")],shell=False)
    
def main():
    realData=Path(f"/mnt/c/Users/ciroj/Desktop/Parquet Files/TnP/MultifileHistograms/AllFilesData/AllRealData.parquet")
    simData=Path(f"/mnt/c/Users/ciroj/Desktop/Parquet Files/TnP/MultifileHistograms/AllFilesData/DY_postEE_2022.parquet")
    blank="0"
    #section=choose_section(blank)
    #axis=choose_axis(blank)
      
    paths= {
        "Real Data":realData,
        "Sim Data":simData }
    for section in ["Barrel","Endcap"]:
            for name,data in paths.items():
                make3DHist(name,data,section)
    """
    for axis in ["probe_pt","probe_superclusterEta"]:
        for section in ["Barrel","Endcap"]:
            for name,data in paths.items():
                makeHist(name,data,section,axis)  
    """
    
main()
SF.conclude()

"""
THE POINT IN QUESTION IS 55.0
Initial Number of Events Inputed8035
Total Signal Events from Integral: 6084
REMOVED EVENTS FROM BACKGROUND: 1951.4024037972367 (24.28627758303966%)

THE POINT IN QUESTION IS 55.0
Initial Number of Events Inputed32371
Total Signal Events from Integral: 7024
REMOVED EVENTS FROM BACKGROUND: 25347.395162941262 (78.30278694801292%)

THE POINT IN QUESTION IS 55.0
Initial Number of Events Inputed8279
Total Signal Events from Integral: 4987
REMOVED EVENTS FROM BACKGROUND: 3292.020597834665 (39.763505228103206%)

"""