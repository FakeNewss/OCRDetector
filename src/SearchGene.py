from __future__ import division
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib as mal
import matplotlib.pyplot as plt
from math import *
x=np.linspace(0,3.14*3,100)
y=np.sin(x) + np.random.normal(loc=0.0,scale=0.1,size=len(x))

# statsmodels.api
import statsmodels.api as sm
lowess=sm.nonparametric.lowess
y_sm=lowess(y,x,frac=0.1)
plt.plot(x,y,lw=1,color='gray',label='y')
plt.plot(y_sm[:,0],y_sm[:,1],lw=1,color='g',label='sm')


# Python seaborn.lmplot()
import seaborn as sns
d=np.hstack((x.reshape(-1,1),y.reshape(-1,1)))
df=DataFrame(d,columns=['xdata','ydata'])
sns.lmplot(x='xdata', y='ydata', data=df,lowess=True) # 实际是调用 statsmodels，且使用默认参数 frac=0.667


# 自定义函数 np.convolve()
def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

y_np3=smooth(y,3) # 点的个数
y_np67=smooth(y,67)
plt.plot(x,y_np3,color='cyan',lw=1)
plt.plot(x,y_np67,color='orange',lw=1)


# Savitzky-Golay filter 平滑
from scipy.signal import savgol_filter
y_sg=savgol_filter(y, 21, 3) # window size 21, polynomial order 3
plt.plot(x,y_sg,color='g',lw=1)


# 周期性信号 傅里叶变换
import scipy.fftpack

N=len(x)
w = scipy.fftpack.rfft(y)
f = scipy.fftpack.rfftfreq(N, x[1]-x[0])
spectrum = w**2

cutoff_idx = spectrum < (spectrum.max()/5)
w2 = w.copy()
w2[cutoff_idx] = 0
y_fft = scipy.fftpack.irfft(w2)
plt.plot(x,y_fft,color='g',lw=1)


# 样条插值
import numpy as np
from scipy.interpolate import splev, splrep
xnew=x
spl=splrep(x,y,k=3)  # 3次
y_spl=splev(xnew,spl)
plt.plot(xnew,y_spl,color='g',lw=1)


# Rbf 插值
from scipy.interpolate import Rbf
xnew=x
rbf=Rbf(x,y)
y_rbf=rbf(xnew)
plt.plot(xnew,y_rbf,color='g',lw=1)


# use fitpack2 method
from scipy.interpolate import Rbf, InterpolatedUnivariateSpline
xnew=x
ius = InterpolatedUnivariateSpline(x, y)
y_ius=rbf(xnew)
plt.plot(xnew,y_ius,color='g',lw=1)


# KernelReg
from statsmodels.nonparametric.kernel_regression import KernelReg

xnew=x
# The third parameter specifies the type of the variable x;
# 'c' stands for continuous;
# 'u' stands for discrete(unordered)
kr = KernelReg(y,x,'c')
y_kr= kr.fit(xnew)[0]
plt.plot(xnew,y_kr,color='g',lw=1)

######################
# plot
fig,ax=plt.subplots(2,5,figsize=(18,9))
ax=ax.flatten()
for a in ax:
    a.set_xticklabels([])
    a.set_yticklabels([])
    a.plot(x,sin(x),lw=1,color='gray')
    #a.plot(x,y,lw=1,color='gray')
    a.set_ylim(-1.1,1.1)

ax[0].plot(x,y_sm[:,1],color='g',lw=1) # statsmodels lowess
ax[1].plot(x,y_np3,color='g',lw=1)     # 自定义函数
ax[2].plot(x,y_np67,color='g',lw=1)    # 自定义函数
ax[3].plot(x,y_sg,color='g',lw=1)     # savgol_filter
ax[4].plot(x,y_fft,color='g',lw=1)    # fft
ax[5].plot(x,y_spl,color='g',lw=1)     # bspline
ax[6].plot(x,y_rbf,color='g',lw=1)    # Rbf
ax[7].plot(x,y_ius,color='g',lw=1)    # ius
ax[8].plot(x,y_kr,color='g',lw=1)   # KernelReg
plt.tight_layout()