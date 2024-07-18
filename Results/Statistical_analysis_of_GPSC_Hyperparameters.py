# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 05:27:14 2024

@author: Nikola Anđelić
"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 

data = pd.read_csv("All_Best_GPSCHYPE_Without_Names.csv")
print(data)
print(data.describe())
# data.describe().to_csv("stat_GPSC_HIP.csv")

data.corr().to_csv("Corr_GPSC_HIP.csv")
plt.rcParams["font.family"] = "Times New Roman"
SMALL_SIZE = 20
MEDIUM_SIZE = 22
BIGGER_SIZE = 24

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

plt.figure(figsize=(25,20))
sns.heatmap(data.corr(),annot=True)
# plt.show()
plt.savefig("GPSC_Hype_Stat.png",
            dpi=300, 
            bbox_inches="tight")