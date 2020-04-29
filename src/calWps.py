import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from  WpsCal import *
import statsmodels.api as sm


def getData(filePath, cnt):
    file = open(filePath)
    indexList = []
    dataList = []
    while True:
        line = file.readline()
        if line == '' or cnt < 0:
            break
        # cnt -= 1
        data =line.split('\t')
        indexList.append(int(data[1]))
        dataList.append(float(data[3][0:-1]))
    return indexList, dataList, indexList[0], indexList[-1]
if __name__ == '__main__':
    # filePathAll = '/mnt/X500/farmers/chenlb/CellData/IH01/IH01PartFragment.bam.bed.count'
    # filePathSub = '/mnt/X500/farmers/chenlb/CellData/IH01/IH01PartFragmentSubSeq.bam.bed.count'
    filePathAll = '/mnt/X500/farmers/luozw/chenlb/fragment.bam.bed.count'
    filePathSub = '/mnt/X500/farmers/luozw/chenlb/fragmentSubSeq.bam.bed.count'
    allIndexList, allList, allStart, allEnd = getData(filePathAll, 10000)
    subIndexList, subList, subStart, subEnd = getData(filePathSub, 10000)
    allArray = np.zeros(allEnd - allStart + 3, dtype=float)
    subArray = np.zeros(allEnd - allStart + 3, dtype=float)
    for i in range(len(allList)):
        index = allIndexList[i]
        allArray[index - allStart] = allList[i]
    for i in range(len(subList)):
        index = subIndexList[i]
        subArray[index - allStart] = subList[i]
    wpsList = np.subtract(subArray, np.subtract(allArray, subArray))
    lowess = sm.nonparametric.lowess
    for beg in range(20000, 38000, 2000):
        e = beg + 2000
        x = np.arange(beg, e, step=1, dtype=int)
        z = lowess(wpsList[beg:e], x, frac=0.04)
        wpsMedian = np.median(wpsList[beg:e])
        smoothData = smooth(wpsList[beg:e])
        wpsListPred = np.subtract(wpsList[beg:e], np.array(z[:, 1])) * 5 + wpsMedian
        z2 = lowess(wpsListPred, x, frac=0.04)
        zAdjust = preprocessing.minmax_scale(z2[:,1])
        medZAdj = np.median(zAdjust)
        zAdjust = zAdjust - medZAdj
        fig, axes = plt.subplots(2, 1, figsize=(10,8))
        axes[0].grid(linestyle='--')
        axes[0].set_title('原始wps曲线和lowess(局部加权回归)处理后的WPS曲线（GC校正步骤)')
        line1 = axes[0].plot(x, wpsList[beg:e], color='#F4700B')
        line2 = axes[0].plot(x, z2[:, 1], color='#1512F8')

        axes[1].grid(linestyle='--')
        axes[1].set_title('lowess(局部加权回归)处理后的WPS曲线（GC校正步骤)')
        # plt.plot(z[:,0], z[:, 1],color='r',lw=1)
        line3 = axes[1].plot(x, zAdjust, color='#1512F8')
        # legend(loc='upper right')
        axes[0].legend((line1[0], line2[0]), ('原始wps曲线', 'lowess(局部加权回归)处理后的WPS曲线'), loc='upper right')
        plt.xlabel('1号染色体基因组位点')
        axes[0].set_ylabel('WPS值')
        axes[1].set_ylabel('标准化WPS值')
        plt.show()
        # filePath = '/mnt/X500/farmers/chenlb/CellData/IH01/IH01Part.bam'
        # # filePath = '/mnt/X500/farmers/luozw/chenlb/xx.bam'
        # start = allStart
        # end = allEnd
        # contig = str(1)
        # bamfile = readFileData(filePath)
        # win = 120
        # # wpsList = wpsCal(bamfile, win, contig, start, end, s, e)
        # # bamfile = readFileData(filePath)
        # wpsList2 = np.array([0 for i in range(allEnd - allStart + 3)])
        # wpsList2 = wpsCal(wpsList2, bamfile, win, contig, start, end, 0, 0)
        # plt.subplot(212)
        # plt.plot(wpsList2[2000:4000])
        # plt.show()

