import pandas as pd
import vector
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import math
pd.set_option("display.max_rows", None)

file = "/mnt/c/Users/ciroj/Desktop/Parquet Files/TnP/DY_postEE_2022/nominal/0a3e7a74-c23e-11ee-aafd-92828e80beef_%2FEvents%3B1_0-163804.parquet"
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


oldval = 0.000
for idx, val in ZM.items(): #filters out sequentially repeated values of ZM
    if val == oldval:
        ZM=ZM.drop(idx)
    oldval=val



plt.hist(ZM, bins=60, range=(0,160),color='red', edgecolor='black')
plt.savefig("hist.png")

#for i in range (1,50):
    #print(ZM.iloc[i])
    #print(tagPt[i+1],probePt[i])


    

