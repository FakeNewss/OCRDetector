#!/usr/bin/python3
#@File: Result.py
#-*-coding:utf-8-*-
#@Author:cuijia

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# import seaborn as sns

data = pd.read_csv("realdata.txt",encoding='gbk',sep='\t')
index = data.Number

df = data.ix[:,1:5]
df.index = index
print(df)

# plt.figure(facecolor='snow',figsize=(16, 8))
df.plot.bar(alpha=0.7,color=['khaki','lightcoral','palegreen','violet'],grid=False,figsize=(15, 8), rot=360, colormap='r')
# df.plot.bar(stacked=True,alpha=0.7,rot=0)
# df.plot(kind='bar',alpha=0.7,color=['blue','orange','green','coral'],grid=False,figsize=(15, 8))
plt.legend(loc=2, bbox_to_anchor=(1.0,1.0),borderaxespad = -0.2)
plt.show()
# plt.savefig('p2.png')
