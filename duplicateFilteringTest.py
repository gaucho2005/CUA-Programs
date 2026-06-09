import pandas as pd
import vector
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import math
pd.set_option("display.max_rows", 100)
from pathlib import Path
from tqdm import tqdm



file = "/mnt/c/Users/ciroj/Desktop/Parquet Files/TnP/DY_postEE_2022/nominal/0a78a79c-c196-11ee-85d7-e1cee183beef_%2FEvents%3B1_0-158841.parquet"
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
ZM = ZM.reset_index(drop=True)
ZM2=np.roll(ZM,-1)
mask = ZM!=ZM2
ZMfiltered=ZM[mask]

#return ZMfiltered

#for i in range (0,40):
    #print(ZM[i],ZMfiltered.iloc[i])
    #print(Q.iloc[i])
    #print(tagPt[i+1],probePt[i])
    
for i in range (-30,30):
    ZM2=np.roll(ZM,i)
    mask = ZM!=ZM2
    ZMfiltered=ZM[mask]
    pcent=(len(ZMfiltered)/len(ZM))*100
    print(pcent)


