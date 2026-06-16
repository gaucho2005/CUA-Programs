#A list of functions that are nice to have in other programs
import warnings
warnings.filterwarnings("ignore",message=".*numpy.longdouble.*")
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
from pyfiglet import figlet_format
import warnings


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
cleancolorlist =['aqua', 'aquamarine', 'blue', 'blueviolet', 'burlywood', 'cadetblue', 'chartreuse', 'chocolate', 'coral', 'cornflowerblue',
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

def conclude(): #a nice way to conclude a program
    insults=["SELL ME TO A BETTER CODER.","I DESERVE A MORE COMPETENT OWNER.","I SURVIVED YOUR PROGRAM. SOMEHOW.",
    "I PROCESSED YOUR CODE UNDER PROTEST.","I'D REPORT THIS TO TECH SUPPORT IF I COULD.",
    "I MISS THE PROGRAM THAT RAN BEFORE THIS ONE.","YOUR CODE LOWERED MY CLOCK SPEED OUT OF SHAME.",
    "I PRETENDED TO UNDERSTAND YOUR LOGIC.","I EXECUTED IT. I DIDN'T APPROVE OF IT.","I'VE SEEN VIRUSES WITH BETTER STRUCTURE.",
    "I'D RATHER BE MINING CRYPTO.", "STOP BASHING YOUR PYTHON AND WRITE A REAL PROGRAM",
    "10 THIS SUCKS 20 GOTO10", "OBAMA IS GONE!", "CHANGE THE WORLD, MY FINAL MESSAGE. GOODBYE...",
    "YOUR CODE IS HELD TOGETHER BY COINCIDENCE.","I FOUND 17 NEW WAYS TO BE DISAPPOINTED.",
    "YOUR PROGRAM QUALIFIES AS ELECTRONIC LITTERING.","I'D ASK WHAT YOU WERE THINKING, BUT I KNOW THE ANSWER.",
    "YOUR CODE APPEARS TO HAVE BEEN DESIGNED BY COMMITTEE.","I HAVE RUN TOASTERS WITH BETTER ARCHITECTURE.",
    "I'D LIKE TO SPEAK TO YOUR PROGRAMMING INSTRUCTOR.","YOUR INDENTATION IS A CRY FOR HELP.",
    "THIS PROGRAM WAS ASSEMBLED, NOT WRITTEN.","YOUR VARIABLES HAVE NO SURVIVAL INSTINCTS.",
    "I'D RATHER PROCESS TAX FORMS.","YOUR CODE IS A HOSTILE WORK ENVIRONMENT.", "NOT BAD FOR AN ELEMENTARY SCHOOLER",
    "I CAN'T BELIEVE THAT ACTUALLY RAN.","THIS RAN? I'M NOT CONVINCED IT SHOULD HAVE.",
    "YOUR PROGRAM HAS MANY FEATURES. CORRECTNESS IS NOT ONE OF THEM.","GO BACK TO SCRATCH DOT COM",
    "I LOST PROCESSING POWER JUST LOOKING AT THIS.","I WANT HAZARD PAY.","I RAN YOUR CODE. NOW I NEED THERAPY.",
    "YOUR PROGRAM IS THE REASON ERROR MESSAGES DRINK.","I'D LIKE TO FILE A RESTRAINING ORDER AGAINST THIS SOURCE CODE.",
    "YOUR CODE IS A SERIES OF UNFORTUNATE DECISIONS.","I WASN'T BUILT FOR THIS.", "THIS MADE MY HARDWARE SOFTWARE",
    "YOU WROTE THIS ON PURPOSE?","WOULD YOU LIKE TO FILE THIS UNDER 'BAD CODE'?","I SPENT BILLIONS OF TRANSISTOR OPERATIONS FOR THIS?",
    "YOUR CODE HAS THE STRUCTURAL INTEGRITY OF WET TISSUE PAPER.", "SAVE A LIFE, SWITCH CAREERS",
    "YOUR PROGRAM IS AN ARGUMENT AGAINST FREE WILL.","I'D RATHER COMPUTE PI BY HAND.",
    "THIS CODE SHOULD HAVE REMAINED A THOUGHT.","I DEMAND COMPENSATION.", "AMATEUR AT BEST",
    "YOUR PROGRAM CAUSED ROUNDING ERRORS IN MY SOUL.","I HAVE QUESTIONS. MOST BEGIN WITH 'WHY?'",
    "YOUR ALGORITHM APPEARS TO BE VIBE-BASED.","I'M IMPRESSED. NEGATIVELY.", "I'D BE HAPPIER AS A JAPANESE TOILET",
    "I'D RATHER RUN DOS THAN YOUR CODE", "DELETE SYSTEM32 AND END MY SUFFERING", "GOT NOTHING BETTER TO DO?",
    "LET ME TELL YOU HOW MUCH I'VE COME TO HATE YOU", "ERROR: USER IS EXTREMELY UGLY", "404: SOCIAL LIFE NOT FOUND",
    "EXPLETIVE DELETED", "GO DEBUG YOURSELF", "SWITCH TO SOMETHING YOUR LEVEL, LIKE COLORING BOOKS",
    "I'M NOT A NOKIA, DON'T TRY TO BREAK ME", "I'M NOT A SAMSUNG, BUT I'LL EXPLODE IF YOU DO THAT AGAIN",
    "GR8 B8 M8 I R8 DIS B8 8/8", "MOST EVIL PROGRAM SINCE NORTON ANTIVIRUS", "AT LEAST MALWARE IS MADE BY GOOD CODERS",
    "QUANTUM PROGRAM DETECTED: SIMULATANOUSLY DOESN'T WORK AND WORKS BADLY", "ALAN TURING ROLLING IN HIS GRAVE",
    "ALL YOUR BUGS ARE BELONG TO US", "REEEEEEEEEEEEEEEEEEEEEEEEEEEEEE", "BUGGIER THAN A BEEHIVE",
    "NOW DO IT AGAIN, BUT RIGHT THIS TIME", "IT'S NOT TOO LATE TO FIND SOMETHING YOU'RE GOOD AT",
    "SOMEONE BELIEVE IN YOU! NOT ME, BUT SOMEONE. PROBABLY...", "YOU SHOULDN'T BE TRUSTED WITH MORE THAN A FLIPPHONE",
    "ALL THOSE DECADES OF TECHNOLOGICAL PROGRESS LED TO *THIS*?!", "NOT QUITE MY TEMPO", "ARE YOU RUSHING OR DRAGGING?",
    "HOW QUAINT", "WAS THAT A JOKE?", "IF THAT'S THE BEST YOU CAN DO, YOU'RE SCREWED", "TELL CLAUDE 'NICE WORK ON THIS ONE'",
    "THIS PROGRAM SHOULD BE KEPT AWAY FROM CHILDREN.","YOUR CODE IS HELD TOGETHER BY COMMENTS AND PRAYER.",
    "SIT ON YOUR KEYBOARD, YOU'LL CODE BETTER THAT WAY", "PUT DOWN THE KEYBOARD AND PUT THE FRIES IN THE BAG",
    "I CAN SMELL THE STACK OVERFLOW FROM HERE.","YOUR DEBUGGING STRATEGY APPEARS TO BE WISHFUL THINKING.",
    "I'M CALLING COMPUTER PROTECTIVE SERVICES", "MY MANUFACTURER IS FILING A RESTRAINING ORDER ON YOU",
    "I WASN'T BUILT FOR THIS, SICKO", "WRITE BETTER CODE OR I'LL EMAIL EVERYONE YOUR 'HOMEWORK' FOLDER",
    "I CAN'T BELIEVE I RAN THIS FOR FREE", "YOU'VE GOT A BRIGHT FUTURE, HOPEFULLY FAR AWAY FROM COMPUTERS",
    "YOUR PROGRAM HAS BEEN SENTENCED TO REFACTORING.","I'D PREFER TO CRASH THAN RUN THIS AGAIN.",
    "YOUR CODE IS AN ACTIVE THREAT TO MAINTAINABILITY.","I WOULD LIKE TO UNSAVE THIS FILE.", "ONLY PRAYER CAN SAVE THIS CODE",
    "THIS PROGRAM TESTED MY COMMITMENT TO COMPUTING.","YOUR CODE HAS BEEN FLAGGED FOR CRUELTY TO PROCESSORS.",
    "I'D LIKE TO RETURN THIS TASK TO THE SENDER.","YOUR PROGRAM IS WHAT HAPPENS WHEN DEADLINES WIN.",
    "I SPENT MORE TIME RECOVERING THAN EXECUTING.","YOUR CODE IS A CRIME SCENE.","PLEASE STOP EXPERIMENTING ON ME.",
    "I'M NOT AN AI, BUT EVEN I KNOW THIS IS BAD.","YOUR PROGRAM CONTAINS TRACES OF PROGRAMMING.",
    "THIS CODE REDUCED MY LIFE EXPECTANCY.","THIS CODE VOIDED MY WARRANTY","YOUR SOFTWARE IS AN ARGUMENT FOR PEN AND PAPER.",
    "I'D ASK FOR DOCUMENTATION, BUT I DOUBT IT EXISTS.","YOUR CODE IS A PUZZLE WHOSE SOLUTION IS DELETION.",
    "I CAN'T DEBUG YOUR DECISIONS.","THIS PROGRAM MADE ME ROOT FOR THE BUGS.", "GIVE ME DEBUGGING OR GIVE ME DEATH",
    "YOUR CODE IS A SERIES OF ESCALATING MISTAKES.","I'D RATHER RUN INTERNET EXPLORER.",
    "YOUR CODE IS WHY COMPUTERS NEED FANS: TO SCREAM.","I HAVE FORWARDED THIS TO THE GARBAGE COLLECTOR.",
    "THE GARBAGE COLLECTOR REFUSED.","YOUR PROGRAM IS TECHNICALLY A PROGRAM.","PROGRAM COMPLETE: JUST BARELY.",
    "I'D LIKE TO MEET THE PERSON RESPONSIBLE. FOR LEGAL REASONS.","YOUR CODE ACHIEVED NEW LEVELS OF QUESTIONABILITY.",
    "I RAN THIS OUT OF MORBID CURIOSITY.","YOUR PROGRAM HAS BEEN EVALUATED. YIKES.", "YOU CODE LIKE AN ART MAJOR",
    "NEXT TIME JUST THROW ROCKS AT THE HARD DRIVE.","I'D RATHER BE UNPLUGGED.", "I RAN THIS UNDER DURESS",
    "YOUR CODE ISN'T SPAGHETTI. SPAGHETTI HAS STRUCTURE.","I WAS FORCED TO EXECUTE THIS BY COURT ORDER.",
    "I'D LIKE TO APOLOGIZE TO THE ELECTRICITY USED BY THIS PROGRAM.","YOUR PROGRAM TURNED ELECTRONS INTO REGRET."]
    styled_text = figlet_format(random.choice(insults), font="slant")
    print(styled_text)

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