#applies some criteria then calculates selection efficiency
"""
In order to test your criteria, you must have a data pool that consists entirely of real electrons,
so you must have already done Z boson mass selection, etc.
Only then can you apply this test criteria, and the efficiency tells you how many real electrons will be able to pass that criteria
"""
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

def calcEfficiency(TagTagList,TagProbeList,TagFailList):
    if not isinstance(TagTagList, (int, float)):
        TT=len(TagTagList)
        TP=len(TagProbeList) #taking number of each pair that exists
        TF=len(TagFailList)
    else:
        TT=TagTagList
        TP=TagProbeList
        TF=TagFailList
    Efficiency=((2*TT)+TP)/((2*TT)+TP+TF)
    return Efficiency