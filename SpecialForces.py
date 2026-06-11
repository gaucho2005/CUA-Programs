#A list of functions that are nice to have in other programs
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
import random

cheatcolumnnames=['tag_seediEtaOriX', 'tag_cutBased', 'tag_electronVeto', 'tag_hasConversionTracks', 'tag_isScEtaEB', 
    'tag_isScEtaEE', 'tag_mvaID_WP80', 'tag_mvaID_WP90', 'tag_pixelSeed', 'tag_seedGain', 'tag_electronIdx', 'tag_jetIdx', 
    'tag_seediPhiOriY', 'tag_vidNestedWPBitmap', 'tag_ecalPFClusterIso', 'tag_energyErr','tag_energyRaw', 'tag_esEffSigmaRR',
    'tag_esEnergyOverRawE', 'tag_eta', 'tag_etaWidth', 'tag_haloTaggerMVAVal', 'tag_hcalPFClusterIso', 'tag_hoe', 'tag_hoe_PUcorr',
    'tag_mvaID', 'tag_pfChargedIso', 'tag_pfChargedIsoPFPV', 'tag_pfChargedIsoWorstVtx', 'tag_pfPhoIso03', 'tag_pfRelIso03_all_quadratic', 
    'tag_pfRelIso03_chg_quadratic', 'tag_phi', 'tag_phiWidth', 'tag_r9', 'tag_s4', 'tag_sieie', 'tag_sieip', 'tag_sipip', 
    'tag_superclusterEta', 'tag_trkSumPtHollowConeDR03', 'tag_trkSumPtSolidConeDR04', 'tag_x_calo', 'tag_y_calo', 'tag_z_calo', 
    'tag_electronIdxG', 'tag_jetIdxG', 'tag_ScEta', 'tag_pt_raw', 'tag_rho_smear', 'tag_pt', 'probe_seediEtaOriX', 'probe_cutBased', 
    'probe_electronVeto','probe_hasConversionTracks', 'probe_isScEtaEB', 'probe_isScEtaEE', 'probe_mvaID_WP80', 'probe_mvaID_WP90',
    'probe_pixelSeed', 'probe_seedGain', 'probe_electronIdx', 'probe_jetIdx', 'probe_seediPhiOriY', 'probe_vidNestedWPBitmap', 
    'probe_ecalPFClusterIso', 'probe_energyErr', 'probe_energyRaw', 'probe_esEffSigmaRR', 'probe_esEnergyOverRawE', 'probe_eta', 
    'probe_etaWidth', 'probe_haloTaggerMVAVal', 'probe_hcalPFClusterIso', 'probe_hoe', 'probe_hoe_PUcorr', 'probe_mvaID', 
    'probe_pfChargedIso', 'probe_pfChargedIsoPFPV', 'probe_pfChargedIsoWorstVtx', 'probe_pfPhoIso03', 'probe_pfRelIso03_all_quadratic',
    'probe_pfRelIso03_chg_quadratic', 'probe_phi', 'probe_phiWidth', 'probe_r9', 'probe_s4','probe_sieie', 'probe_sieip', 
    'probe_sipip', 'probe_superclusterEta', 'probe_trkSumPtHollowConeDR03', 'probe_trkSumPtSolidConeDR04', 'probe_x_calo', 
    'probe_y_calo', 'probe_z_calo', 'probe_electronIdxG', 'probe_jetIdxG', 'probe_ScEta', 'probe_pt_raw', 'probe_rho_smear', 
    'probe_pt', 'sigma_m_over_m', 'sigma_m_over_m_Smeared', 'mass', 'nPV', 'fixedGridRhoAll']
cleancolorlist =['aqua', 'aquamarine', 'blue', 'blueviolet','brown', 'burlywood', 'cadetblue', 'chartreuse', 'chocolate', 'coral', 'cornflowerblue',
    'crimson', 'cyan', 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray','darkgreen', 'darkgrey', 'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange',
    'darkorchid', 'darkred', 'darksalmon', 'darkseagreen', 'darkslateblue', 'darkslategray', 'darkslategrey', 'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue', 'dimgray',
    'dimgrey', 'dodgerblue', 'firebrick', 'forestgreen', 'fuchsia', 'gold', 'goldenrod', 'gray', 'green', 'greenyellow', 'grey', 'hotpink', 'indianred', 'indigo', 'khaki',
    'lightblue', 'lightcoral', 'lightcyan', 'lightgray', 'lightgreen', 'lightgrey', 'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightslategray','lightslategrey', 'lightsteelblue', 'lime', 'limegreen', 'magenta', 'maroon',
    'mediumaquamarine', 'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen', 'mediumslateblue', 'mediumspringgreen', 'mediumturquoise', 'mediumvioletred',
    'midnightblue', 'navy', 'olive', 'olivedrab', 'orange', 'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise', 'palevioletred', 'peachpuff',
    'peru', 'pink', 'plum', 'powderblue', 'purple', 'rebeccapurple', 'red', 'rosybrown', 'royalblue', 'saddlebrown', 'salmon', 'sandybrown', 'seagreen', 'sienna', 'silver',
    'skyblue', 'slateblue', 'slategray', 'slategrey', 'springgreen', 'steelblue', 'tan', 'teal', 'thistle', 'tomato', 'turquoise', 'violet', 'wheat', 'yellow', 'yellowgreen']
def randColor():
    return random.choice(cleancolorlist)
def get_column_arrays(target_col, file_path): #Fast. Grabs the data from a column in a file at one file path. You can easily repeat this and loop to get a dictionary of columns and their data
    
    
    p=Path(file_path)
    
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
    column = df[target_col].to_numpy().astype("float64")
        
    return column #returns column data as NumPy array
    

def FillLHCDictionary(emptydict,datapath):  #feed it an empty dictionary and a LHC parquet filepath, and it'll fill that dictionary with each column in the parquet file
    for i in tqdm(cheatcolumnnames,desc="Collecting Data Columns"):
        emptydict[i]= get_column_arrays(i,datapath)
    return(emptydict) #returns dictionary, now full


def save_color_swatch(color_list, outfolder): #save color swatches from a list of colors
    os.makedirs(outfolder, exist_ok=True)

    for i, color in enumerate(color_list):
        fig, ax = plt.subplots(figsize=(2, 2))

        # white background
        ax.set_facecolor("white")
        fig.patch.set_facecolor("white")

        # draw a rectangle filled with the color
        ax.add_patch(
            plt.Rectangle((0, 0), 1, 1, color=color)
        )

        # remove axes for a clean swatch
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # save file
        filename = f"color_{i:03d}_{color.replace('#', '')}.png"
        filepath = os.path.join(outfolder, filename)
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close(fig)