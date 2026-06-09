#should probably try making all my steps functions then just repeating the functions for each file? 
#make histograms out of a file full of column files, applies some filtering
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
import pyarrow.parquet as pq
import polars as pl
import random
color_palette = [
    'red', 'blue', 'green', 'orange', 'purple', 'brown', 
    'cyan', 'magenta', 'lime', 'navy', 'teal', 'gold'
]
css4_colors = [
    'aliceblue', 'aqua', 'aquamarine', 'azure', 'beige', 'bisque', 'black', 'blanchedalmond', 'blue', 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse',
    'chocolate', 'coral', 'cornflowerblue', 'cornsilk', 'crimson', 'cyan', 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen', 'darkgrey', 'darkkhaki', 'darkmagenta', 'darkolivegreen','darkorange', 'darkorchid', 'darkred', 'darksalmon', 'darkseagreen',
    'darkslateblue', 'darkslategray', 'darkslategrey', 'darkturquoise', 'darkviolet','deeppink', 'deepskyblue', 'dimgray', 'dimgrey', 'dodgerblue','firebrick', 'forestgreen', 'fuchsia', 'gainsboro',
    'gold', 'goldenrod', 'gray', 'green','greenyellow', 'grey', 'honeydew', 'hotpink', 'indianred','indigo', 'ivory', 'khaki', 'lavender', 'lavenderblush','lawngreen', 'lemonchiffon', 'lightblue', 'lightcoral', 'lightcyan',
    'lightgoldenrodyellow', 'lightgray', 'lightgreen', 'lightgrey', 'lightpink','lightsalmon', 'lightseagreen', 'lightskyblue', 'lightslategray', 'lightslategrey','lightsteelblue', 'lightyellow', 'lime', 'limegreen', 'linen','magenta', 'maroon', 'mediumaquamarine', 'mediumblue', 'mediumorchid','mediumpurple', 'mediumseagreen', 'mediumslateblue', 'mediumspringgreen', 'mediumturquoise',
    'mediumvioletred', 'midnightblue', 'mintcream', 'mistyrose', 'moccasin', 'navy', 'oldlace', 'olive', 'olivedrab','orange', 'orangered', 'orchid', 'palegoldenrod', 'palegreen','paleturquoise', 'palevioletred', 'papayawhip', 'peachpuff', 'peru',
    'pink', 'plum', 'powderblue', 'purple', 'rebeccapurple','red', 'rosybrown', 'royalblue', 'saddlebrown', 'salmon','sandybrown', 'seagreen', 'seashell', 'sienna', 'silver','skyblue', 'slateblue', 'slategray', 'slategrey', 'snow','springgreen', 'steelblue', 'tan', 'teal', 'thistle',
    'tomato', 'turquoise', 'violet', 'wheat', 'yellow', 'yellowgreen'
]
cheatcolumnnames=['tag_seediEtaOriX', 'tag_cutBased', 'tag_electronVeto', 'tag_hasConversionTracks', 'tag_isScEtaEB', 'tag_isScEtaEE', 'tag_mvaID_WP80', 'tag_mvaID_WP90', 'tag_pixelSeed', 'tag_seedGain', 'tag_electronIdx', 'tag_jetIdx', 'tag_seediPhiOriY', 'tag_vidNestedWPBitmap', 'tag_ecalPFClusterIso', 'tag_energyErr',
 'tag_energyRaw', 'tag_esEffSigmaRR', 'tag_esEnergyOverRawE', 'tag_eta', 'tag_etaWidth', 'tag_haloTaggerMVAVal', 'tag_hcalPFClusterIso', 'tag_hoe', 'tag_hoe_PUcorr', 'tag_mvaID', 'tag_pfChargedIso', 'tag_pfChargedIsoPFPV', 'tag_pfChargedIsoWorstVtx', 'tag_pfPhoIso03', 'tag_pfRelIso03_all_quadratic', 'tag_pfRelIso03_chg_quadratic', 'tag_phi', 'tag_phiWidth', 'tag_r9', 'tag_s4',
 'tag_sieie', 'tag_sieip', 'tag_sipip', 'tag_superclusterEta', 'tag_trkSumPtHollowConeDR03', 'tag_trkSumPtSolidConeDR04', 'tag_x_calo', 'tag_y_calo', 'tag_z_calo', 'tag_electronIdxG', 'tag_jetIdxG', 'tag_ScEta', 'tag_pt_raw', 'tag_rho_smear', 'tag_pt', 'probe_seediEtaOriX', 'probe_cutBased', 'probe_electronVeto',
 'probe_hasConversionTracks', 'probe_isScEtaEB', 'probe_isScEtaEE', 'probe_mvaID_WP80', 'probe_mvaID_WP90','probe_pixelSeed', 'probe_seedGain', 'probe_electronIdx', 'probe_jetIdx', 'probe_seediPhiOriY', 'probe_vidNestedWPBitmap', 'probe_ecalPFClusterIso', 'probe_energyErr', 'probe_energyRaw', 'probe_esEffSigmaRR', 
 'probe_esEnergyOverRawE', 'probe_eta', 'probe_etaWidth', 'probe_haloTaggerMVAVal', 'probe_hcalPFClusterIso', 'probe_hoe', 'probe_hoe_PUcorr', 'probe_mvaID', 'probe_pfChargedIso', 'probe_pfChargedIsoPFPV', 'probe_pfChargedIsoWorstVtx', 'probe_pfPhoIso03', 'probe_pfRelIso03_all_quadratic', 'probe_pfRelIso03_chg_quadratic', 'probe_phi', 'probe_phiWidth', 'probe_r9', 'probe_s4',
 'probe_sieie', 'probe_sieip', 'probe_sipip', 'probe_superclusterEta', 'probe_trkSumPtHollowConeDR03', 'probe_trkSumPtSolidConeDR04', 'probe_x_calo', 'probe_y_calo', 'probe_z_calo', 'probe_electronIdxG', 'probe_jetIdxG', 'probe_ScEta', 'probe_pt_raw', 'probe_rho_smear', 'probe_pt', 'sigma_m_over_m', 'sigma_m_over_m_Smeared', 'mass', 'nPV', 'fixedGridRhoAll']
skippedtotal=0
def skip_column(col):
    # skip booleans
    if col[1].dtype == bool or col[1].dtype == 'bool': #skips the boolean columns
        print("BOOLBOOLBOOLBOOLBOOLBOOLBOOLBOOLBOOLBOOLBOOL")
        return True

    # skip columns of less than 1000 unique values
    if np.unique(col).size <= 1000:
        print("UNIQUEUNIQUEUNIQUEUNIQUEUNIQUEUNIQUEUNIQUEUNIQUEUNIQUEUNIQUEUNIQUEUNIQUE")
        
        return True
    else:
        return False

filesfolder = Path(f"/mnt/c/Users/ciroj/Desktop/Parquet Files/TnP/FourFileHistograms/AllFilesData")
folderfiles = list(filesfolder.iterdir())
out_folder = Path("/mnt/c/Users/ciroj/Desktop/Parquet Files/TnP/FourFileHistograms")

 

totalnans=0
def genHist(colname,col): #at present moment, generates a histogram for each column given
    nan_count = np.isnan(col).sum()
    #print(f"Found and removed {nan_count} NaN values.")
    total=totalnans+nan_count
    col = col[np.isfinite(col)] #finds and removes nans
    
    counts, bin_edges = np.histogram(col, bins=200)
    """
    if counts.max()>1100000: #implements different rules for histograms that would have an excessive amount of data concentrated in one bin
        counts, bin_edges = np.histogram(col, bins=350) #was 300
        
        indices = np.argsort(-counts) #if the highest bar is more than double the second highest, cut it off
        sorted_values = counts[indices]
        if sorted_values[0]>(2*sorted_values[1]): 
            plt.ylim(top=(1.1*sorted_values[1]),bottom=0)
            mask= counts>=(.005*sorted_values[1])
            filtered_counts = counts[mask]
        else:
            mask= counts>=(.005*sorted_values[0])
            filtered_counts = counts[mask]
           
        

    else:
        mask = counts >= 1000
        filtered_counts = counts[mask] #filters out bars with less than a certain number of entries
    """
    mask = counts >= 1000
    filtered_counts = counts[mask] #filters out bars with less than a certain number of entries
    #print(f"Max bar is {filtered_counts.max()}")
    filtered_edges = bin_edges[:-1][mask]
    width = np.diff(bin_edges)[0]
    
    filtered_counts = filtered_counts[:-1] #TEMPORARY FIX, IF THIS WORKS DO NOT KEEP IT FOREVER BECAUSE THE PROGRAM WILL NOT BE COMPLETE REEEEEEEEEEEE
    total_area = sum(filtered_counts) * width
    density_values = [count / total_area for count in filtered_counts]
    plt.bar(filtered_edges[:-1], density_values, width=width, align='edge',color=random.choice(css4_colors), alpha=0.5, label=colname)
    
    #plt.bar(filtered_edges, filtered_counts, density=True, width=width, color=random.choice(css4_colors),alpha=0.5,label=colname)
    plt.ticklabel_format(style='plain', axis='y')


def get_column_arrays(target_col, file_paths):
    """
    Efficiently extracts a single column from multiple Parquet files.
    Returns a dictionary of 1D numpy arrays (key = filename, value = 1D array).
    """
    column_data = {}
    
    for path in file_paths:
        p = Path(path)
        
        # 1. Lazy scan: Only look at metadata
        # 2. Select: Only target the specific column (ignores the other 117)
        # 3. Collect (streaming): Read the column in chunks to save RAM
        df = (
            pl.scan_parquet(p)
            .select(target_col)
            .collect()
        )
        
        # Extract as a 1D NumPy array and store it in the dictionary
        # NumPy arrays are much more memory efficient than standard Python lists
        column_data[p.name] = df[target_col].to_numpy().astype("float64")
        
    return column_data
    
"""
if skip_column(columnname): #skips invalid columns
    print("skipping invalid column (contains booleans or too few unique values)")
    global skippedtotal
    skippedtotal+=1
    continue
"""
        

        
        
        
def grab4columns(target):
    # How to use it in your workflow

    # 1. Define your paths (using WSL format)
    base_dir = "/mnt/c/Users/ciroj/Desktop/Parquet Files/TnP/FourFileHistograms/AllFilesData/"
    my_files = [
        base_dir + "DataE_2022.parquet",
        base_dir + "DataF_2022.parquet", # Replace with actual names
        base_dir + "DataG_2022.parquet",
        base_dir + "DY_postEE_2022.parquet"
    ]

    # 3. Call the bridge function!
    hist_data = get_column_arrays(target, my_files)

    # 4. Access your separate 1D arrays for plotting
    global data_file_A
    global data_file_B
    global data_file_C
    global data_file_D
    data_file_A = hist_data["DataE_2022.parquet"]
    data_file_B = hist_data["DataF_2022.parquet"]
    data_file_C = hist_data["DataG_2022.parquet"]
    data_file_D = hist_data["DY_postEE_2022.parquet"]
    global my_datasets
    my_datasets = {
    "data_file_A": data_file_A,
    "data_file_B": data_file_B,
    "data_file_C": data_file_C,
    "data_file_D": data_file_D
}

def main():
    
    #folder = Path(f"/mnt/c/Users/ciroj/Desktop/Parquet Files/TnP/{k}/nominal/Columns")
    #parquet_filelist = list(k.glob("*.parquet"))
    print("VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV OUTPUT BEGINS HERE VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV")
    for i in tqdm(cheatcolumnnames,desc="Generating Histograms"):
        plt.close()
        global tarjet
        tarjet=i
        grab4columns(tarjet)
        gennedsomething=0  
        for name,dataset in (my_datasets.items()):
            if skip_column(dataset):    
                print("skipping invalid column (contains booleans or too few unique values)")
                global skippedtotal
                skippedtotal+=.25
                continue
            genHist(name,dataset)
            gennedsomething+=1
        if gennedsomething!=0: #confirms that there is actually data to generate a histogram with, which won't be the case if a column was skipped for all datasets
            plt.legend()
            filename = f"{tarjet}_hist.png"
            plt.savefig(out_folder / filename)
        
    totalnans=0
    print(f"found total {totalnans} nans")
    print(f"skipped {skippedtotal} columns for being boring") 

main()        
    