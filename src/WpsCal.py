import sys
import pysam
import time
import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
from pylab import *
import pandas as pd
import scipy.signal
import numpy as np
import pylab as pl
from scipy.signal import *
from scipy.signal import savgol_filter
from scipy import interpolate
from scipy import integrate
# from pykalman import *
from scipy.signal import medfilt
from sklearn import *
from sklearn.neighbors import KernelDensity
import peakutils
# import pywt
import gc

# import lowess as lo

from SegmentTree import SegmentTree
from NDR import NDR
from Peak import Peak
import Kalman_filter
import callNDR
# import statsmodels.api as sm
# sys.setrecursionlimit(100000)     设置程序最大运行深度
from matplotlib.font_manager import FontProperties
# from numba import jit


font = FontProperties(
    fname='/home/chenlb/anaconda3/lib/python3.7/site-packages/matplotlib/mpl-data/fonts/ttf/SimHei.ttf')

#
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# mpl.rcParams['font.sans-serif'] = ['SimHei']


# path /home/chenlb/WPSCalProject/00.panel_data/
# BAM data:1_1_A00253:147:HGGW2DMXX:2:1170:16839:5916	147	13	32910375	60	94M	=	32910293
# -176	TTTGTCACTTTGTGTTTTTATGTTTAGGTTTATTGCATTCTTCTGTGAAAAGAAGCTGTTCACAGAATGATTCTGAAGAACCAACTTTGTCCTT
# DDFEBHHICCDGEGDCCCCEEFECCHFGDCCDEEHJECEICHIEGEHEEEHHEHHIEGDEKHJHHEEEHICEIEHFHHEDFIEDIBBDFCCD>A
# NM:i:0	MD:Z:94	AS:i:94	XS:i:0	RG:Z:cancer	MC:Z:94M      MQ:i:60

def readFileData(filePath):
    '''
    :param filePath: bam文件路径
    :return: bamfile bam文件
    '''
    # data.template_length  fragment length
    # data.reference_start  start point
    # data.query_name
    # data.query_sequence sequencing data
    bamfile = pysam.AlignmentFile(filePath, 'rb')
    print('get bamfile done')
    return bamfile


def getBamFileInformation(contig, bamfile, start, end):
    '''
    得到bam文件的基础信息
    :param bamfile:bam文件操作符
    :return:start: bam中的reads最小的起点
            end: 文件中最远的fragment终点
    '''
    s = 100000000000;
    e = 0
    # cnt = 0
    for data in bamfile.fetch(contig=contig, start=start, end=end):
        # if cnt > 20000:
        #     break
        # cnt += 1
        if data.isize > 0 and data.isize < 1000:
            s = min(s, data.reference_start)
            e = max(e, data.reference_start + data.isize)
    print('getBamFileInformation done')
    if s == 100000000000 and e == 0:
        return 0, 0
    return s, e

##@jit
def getBamFileInformationByNoOrderBam(contig, bamfile, start, end):
    '''
    得到bam文件的基础信息
    :param bamfile:bam文件操作符
    :return:start: bam中的reads最小的起点
            end: 文件中最远的fragment终点
    '''
    s = 100000000000;
    e = 0
    cnt = 0
    for data in bamfile:
        if data.reference_start >= start and data.reference_start <= end:
            if data.isize > 0 and data.isize < 1000:
                s = min(s, data.reference_start)
                e = max(e, data.reference_start + data.isize)
    print('getBamFileInformation done')
    return s, e

#@jit
def wpsCalBySegTreeByNoOrderBam(bamfile, win, contig, start, end, s, e):
    '''
    利用线段树数据结构加速计算wps值
    :param bamfile: bam文件操作符
    :param win: 计算wps的窗口值
    :param contig: 需计算的染色体名称
    :param start: 染色体提取起点
    :param end: 染色体提取终点
    :param s: 需计算的起点
    :param e: 需计算的终点
    :return: wpsList: wps列表
    '''
    cnt = 0
    wpsList = [0 for i in range(e - s + 1000)]
    print('start : ', start, ' end : ', end)
    wpsSegTree = SegmentTree(len(wpsList), wpsList)
    for data in bamfile:
        if data.reference_start >= start and data.reference_start <= end:
            # print(data.template_length,data.reference_start,' -- ',data.query_name ,data.query_sequence)
            if data.isize > 0 and data.isize < 1000:
                listIndex = data.reference_start - s
                biWin = int(win / 2)
                wpsSegTree.addupdate(1, listIndex + biWin, listIndex + data.isize - biWin, 1, len(wpsList), 1)
                wpsSegTree.addupdate(1, listIndex, listIndex + biWin - 1, 1, len(wpsList), -1)
                wpsSegTree.addupdate(1, listIndex + data.isize - biWin + 1, listIndex + data.isize, 1, len(wpsList), -1)
        else:
            if cnt > 2000:
                break
            cnt += 1
    for i in range(len(wpsList)):
        wpsList[i] = wpsSegTree.query(1, i + 1, i + 1, 1, len(wpsList))
    print('wpsCalBySegTree done')
    return wpsList

#@jit
def wpsCal(wpsList, bamfile, win, contig, start, end, s, e):
    #wpsSegTree, wpsList, bamfile, win, contig, start, end, s, e
    '''
    计算wps值
    :param bamfile: bam文件操作符
    :param win: 计算wps的窗口值
    :param chr: 需计算的染色体名称
    :param start: 需计算的起点
    :param end: 需计算的终点
    :return: wpsList: wps列表
    '''
    cnt = 0
    for data in bamfile.fetch(contig=contig, start=start, end=end):
        # cnt += 1
        # print(data.template_length,data.reference_start,' -- ',data.query_name ,data.query_sequence)
        if data.reference_start - start >= 0 and data.reference_start - start + data.isize < len(
                wpsList) and data.isize > 0 and data.isize < 200 and data.isize > 100:
            listIndex = data.reference_start - start
            biWin = int(win / 2)
            for index in range(listIndex, listIndex + data.isize):
                if index >= listIndex + biWin and index <= listIndex + data.isize - biWin:
                    wpsList[index] += 1
                else:
                    wpsList[index] -= 1
    print('wps calculation done')
    return wpsList

#@jit
def wpsCalBySegTree(wpsSegTree, wpsList, bamfile, win, contig, start, end, s, e):
    '''
    利用线段树数据结构加速计算wps值
    :param bamfile: bam文件操作符
    :param win: 计算wps的窗口值
    :param contig: 需计算的染色体名称
    :param start: 染色体提取起点
    :param end: 染色体提取终点
    :param s: 需计算的起点
    :param e: 需计算的终点
    :return: wpsList: wps列表
    '''
    cnt = 0
    length = len(wpsList) + 4000
    up = [0.0] * length
    down = [0.0] * length

    print('start : ', start, ' end : ', end)
    for data in bamfile.fetch(contig=contig, start=start, end=end):
        # if cnt > 20000:
        #     break
        cnt += 1
        # print(data.template_length,data.reference_start,' -- ',data.query_name ,data.query_sequence)
        if data.reference_start - start >= 0 and data.reference_start - start + data.isize < len(
                wpsList) and data.isize > 0 and data.isize < 1000:

            # ocf
            # print(data.reference_start - start)
            if data.reference_start - start >= 0:
                up[data.reference_start - start] += 1
            if data.reference_start + data.isize - end < length:
                down[data.reference_start + data.isize - end] += 1

            listIndex = data.reference_start - start
            biWin = int(win / 2)
            wpsSegTree.addupdate(1, listIndex + biWin, listIndex + data.isize - biWin, 1, len(wpsList), 1)
            wpsSegTree.addupdate(1, listIndex, listIndex + biWin - 1, 1, len(wpsList), -1)
            wpsSegTree.addupdate(1, listIndex + data.isize - biWin + 1, listIndex + data.isize, 1, len(wpsList), -1)
    for i in range(len(wpsList)):
        wpsList[i] = wpsSegTree.query(1, i + 1, i + 1, 1, len(wpsList))
    print('wpsCalBySegTree done')
    # print(cnt)
    return wpsList, up, down


def callOCF(up, down):
    win = 10
    loc = 60
    length = len(up)
    ocf = [0] * length
    start = loc + win + 1
    end = length - start
    while start <= end:
        tmpocf = 0
        x = start - loc - win
        y = start - loc + win
        while x <= y:
            tmpocf += down[x] - up[x]
            x += 1
        x = start + loc - win
        y = start + loc + win
        while x <= y:
            tmpocf += up[x] - down[x]
            x += 1
        ocf[start] = tmpocf
        start += 1
    return ocf




def draw(dataList):
    '''
    绘制折线图像
    :param dataList: wpsList数据列表
    :return: none
    '''
    colorsList = ['b', 'r', 'k', 'g', 'c', 'm', 'y', 'w']
    plt.title('WPS')
    for i in range(len(dataList)):
        list = dataList[i]
        x = [i for i in range(len(list))]
        plt.plot(x, list[0:len(list)], mec=colorsList[i % len(colorsList)])
    plt.show()
    print('draw done')


def drawPeaks(dataList, win, x, start):
    '''
    :param dataList: 数据列表，每一个子列表data包括三个子列表，
                            wpsFilterData = data[0]
                            peaksX = data[1]
                            properties = data[2]
    :param win: 窗口大小
    :param x: 横坐标
    :param s: 起始位置
    :return: None
    '''
    colorsList = ['b', 'r', 'k', 'g', 'c', 'm', 'y', 'w']
    lineList = []
    cIndex = 0
    fig, axes = plt.subplots(len(dataList), 1, figsize=(20, 4))
    for data in dataList:
        wpsFilterData = np.array(data[0])
        peaksX = data[1]
        peakObjectList = data[2]
        # properties = data[2]
        s_Offset = np.array([start for i in range(len(peaksX))])
        # for peak in peaksX:
        #     plt.vlines(peak - 85, -0.1 - cIndex/10, 0.8 + cIndex/10, colors=colorsList[cIndex], linestyles='dashed')
        #     plt.vlines(peak + 85, -0.1 - cIndex/10, 0.8 + cIndex/10, colors=colorsList[cIndex], linestyles='dashed')
        minLen = min(len(x), len(wpsFilterData))
        print('minLen = ', minLen)
        peaksCorrectionX = np.add(peaksX, s_Offset)
        line = axes[cIndex].plot(x[0: minLen], wpsFilterData[0: minLen], colorsList[cIndex])
        lineList.append(line)
        axes[cIndex].plot(peaksCorrectionX, wpsFilterData[peaksX],
                          'x' + colorsList[(cIndex + 2) % len(colorsList)])
        # axes[cIndex].vlines(x=peaksCorrectionX,
        #                     ymin=np.array(wpsFilterData)[peaksX] - np.array(wpsFilterData)[peaksX] * 1.3,
        #                     ymax=np.array(wpsFilterData)[peaksX] + np.array(wpsFilterData)[peaksX] * 1.3,
        #                     color=colorsList[cIndex])
        # plt.hlines(y=properties['width_heights'], xmin=np.add(properties['left_ips'], s_Offset),
        #            xmax=np.add(properties['right_ips'], s_Offset), color=colorsList[cIndex])
        vX = []
        for peak in peakObjectList:
            vX.append(peak.startPos)
            # print(peak.endPos)
            # if peak.endPos > len(wpsFilterData):
            #     vX.append(len(wpsFilterData) - 1)
            # else:
            if not vX.__contains__(peak.endPos):
                vX.append(peak.endPos)
        vX = np.array(vX)
        print(vX)
        if len(vX) == 0:
            return

        axes[cIndex].vlines(x=vX + start,
                            ymin=wpsFilterData[vX] - abs(wpsFilterData[vX]) * 1.2,
                            ymax=wpsFilterData[vX] + abs(wpsFilterData[vX]) * 1.2, color='k',
                            linestyles='dashed')
        # axes[cIndex].vlines(peaksCorrectionLeft,
        #                     ymin=np.array(wpsFilterData)[leftValleyList] - abs(
        #                         np.array(wpsFilterData)[leftValleyList]),
        #                     ymax=np.array(wpsFilterData)[leftValleyList] + abs(
        #                         np.array(wpsFilterData)[leftValleyList]), color='k', linestyles='dashed')
        # axes[cIndex].vlines(peaksCorrectionRight,
        #                     ymin=np.array(wpsFilterData)[rightValleyList] - abs(
        #                         np.array(wpsFilterData)[rightValleyList]),
        #                     ymax=np.array(wpsFilterData)[rightValleyList] + abs(
        #                         np.array(wpsFilterData)[rightValleyList]), color='blue'
        #                     , linestyles='dashed')
        cIndex = (cIndex + 1) % len(colorsList)
        axes[cIndex].grid(axis="y", linestyle='--')
    plt.legend((lineList[0][0], lineList[1][0]), ('平滑后的WPS曲线', '原始WPS曲线'), loc='lower right')
    plt.show()


def drawSinglePeak(wpsList, rawDataList, peakObjectList, s):
    for peak in peakObjectList:
        line = [peak.startPos, peak.endPos]
        # print(line)
        end = peak.endPos + 550
        if end > len(wpsList):
            end = len(wpsList)
        start = peak.startPos
        x = np.array([s + i for i in range(start, end)])
        fig, axes = plt.subplots(2, 1)
        axes[0].plot(s + peak.peakIndex, wpsList[peak.peakIndex], 'xr')
        axes[0].plot(x, np.array(wpsList)[start:end], color='r')
        axes[1].plot(x, np.array(rawDataList)[start:end], color='b')
        axes[1].plot(s + peak.peakIndex, rawDataList[peak.peakIndex], 'xb')
        axes[0].vlines(x=np.add(np.array([s for i in range(len(line))]), np.array(line)),
                       ymin=wpsList[line] - abs(wpsList[line]) * 1.3,
                       ymax=wpsList[line] + abs(wpsList[line]) * 1.3, color='k', linestyles='dashed')
        axes[1].vlines(x=np.add(np.array([s for i in range(len(line))]), np.array(line)),
                       ymin=np.array(rawDataList)[line] - abs(np.array(rawDataList)[line]) * 1.3,
                       ymax=np.array(rawDataList)[line] + abs(np.array(rawDataList)[line]) * 1.3, color='k',
                       linestyles='dashed')
    plt.show()


def drawPeaksBycwt(dataList, win, x, s):
    '''
    :param dataList: 数据列表，每一个子列表data包括三个子列表，
                            wpsFilterData = data[0]
                            peaksX = data[1]
    :param win: 窗口大小
    :param x: 横坐标
    :param s: 起始位置
    :return: None
    '''
    colorsList = ['b', 'r', 'k', 'g', 'c', 'm', 'y', 'w']

    cIndex = 0
    for data in dataList:
        wpsFilterData = data[0]
        peaksX = data[1]
        s_Offset = np.array([s for i in range(len(peaksX))])
        # plt.vlines(peaksX, wpsFilterData[peaksX] -0.1 - cIndex/10, wpsFilterData[peaksX] + 0.1 + cIndex/10, colors=colorsList[cIndex], linestyles='dashed')
        # plt.vlines(peaksX + 85, wpsFilterData[peaksX] -0.1 - cIndex/10, wpsFilterData[peaksX] + 0.1 + cIndex/10, colors=colorsList[cIndex], linestyles='dashed')
        minLen = min(len(x), len(wpsFilterData))
        print('minLen = ', minLen)
        peaksCorrectionX = np.add(peaksX, s_Offset)
        plt.plot(x[0:minLen], wpsFilterData[0:minLen], colorsList[cIndex])
        plt.plot(peaksCorrectionX, wpsFilterData[peaksX], 'x' + colorsList[cIndex])
        cIndex = (cIndex + 1) % len(colorsList)
    plt.show()


def drawNDR(ndrInformation, ndrObjectList, x, s, rawDataList, smoothData):
    '''
    :param ndrInformation: NDR信息
    :param ndrObjectList: ndr列表
    :param rawData: 原始数据
    :param smoothData: 平滑处理后的数据
    :return:
    '''
    # start = ndrInformation['startPos']
    # end = ndrInformation['endPos']
    if len(ndrObjectList) == 0:
        return
    print('len(ndrObjectList)', len(ndrObjectList))
    for ndr in ndrObjectList:
        print(ndr)
        drawSingleNDR(smoothData, rawDataList, x, ndr)


def drawSingleNDR(smoothData, rawDataList, x, contig, start, ndr):
    fig = plt.figure(figsize=(10, 4))
    axe1 = fig.add_subplot(2, 1, 1)
    axe2 = fig.add_subplot(2, 1, 2)
    xStart = max(ndr.startPos - 2000, 0)
    xEnd = min(ndr.endPos + 2000, min(len(x), len(smoothData)))
    print('xStart : ', xStart, ' xEnd : ', xEnd)
    yminPos = min(smoothData[ndr.startPos], smoothData[ndr.endPos])
    ymaxPos = min(smoothData[ndr.startPos], smoothData[ndr.endPos])
    yminPosRawData = min(rawDataList[ndr.startPos], rawDataList[ndr.endPos])
    ymaxPosRawData = max(rawDataList[ndr.startPos], rawDataList[ndr.endPos])
    line1 = axe1.plot(x[xStart:xEnd], smoothData[xStart:xEnd], 'r', label='平滑后的WPS曲线')
    line2 = axe2.plot(x[xStart:xEnd], rawDataList[xStart:xEnd], 'b', label='原始WPS曲线')

    corX = np.array([start + ndr.startPos, start + ndr.endPos])
    print('corx : ', corX)
    print(x[xStart:xEnd])
    axe1.vlines(corX, ymin=yminPos - 0.5, ymax=ymaxPos + 0.5, color='k')
    axe2.vlines(corX, ymin=yminPosRawData - 30,
                ymax=ymaxPosRawData + 30, color='k')
    plt.legend((line1[0], line2[0]), ('平滑后的WPS曲线', '原始WPS曲线'), loc='lower right')
    title('疑似NDR区域')
    plt.xlabel(str(contig) + '号染色体基因组位点')
    plt.ylabel('wps值')
    plt.pause(0.01)
    plt.show()
    print('drawSingleNDR done')

def writToExcel(dataList):
    '''
    将数据写入excel文件
    :param wpsList: wpsList数据
    :return: none
    '''
    for i in range(len(dataList)):
        dataList[i] = dataList[i].tolist()
        cnt = 1
        for j in range(len(dataList[i]) - 1, 0, -1):
            if (abs(dataList[i][j]) > 0.05):
                break;
            cnt += 1
        dataList[i] = dataList[i][0:len(dataList[i]) - cnt]
    wpsFile = pd.DataFrame(dataList)
    writer = pd.ExcelWriter('test.xlsx')
    wpsFile.to_excel(writer, sheet_name='Data1', startcol=0, index=False)
    writer.save()
    print('write to excel done')


def writeToNDRFile(ndrInformation):
    writer = pd.ExcelWriter('NDR Information.xlsx')
    ndrInformation.to_excel(writer, sheet_name='Data1', startcol=0, index=False)
    writer.save()
    print('write NDR Information to excel done')


def savgol_filter_func(wpsList, filterWin, poly):
    '''
    SG滤波算法平滑波形,多项式平滑算法 Savitzky-Golay平滑算法
    :param wpsList: wpsList数据
    :return: wpsListFiler SG滤波之后的WPS数据
    '''

    x = [i for i in range(len(wpsList))]
    x = np.array(x)  # list to ndarray
    wpsList = np.array(wpsList)
    wpsFilter = savgol_filter(wpsList, filterWin, poly)  # windows length(int)： (must be a positive odd integer);
    # polyorder(int)；The order of the polynomial used to fit the samples. polyorder must be less than window_length.
    print('savgol filter done')
    return wpsFilter


def normalized(wpsList, size):
    wpsArray = np.array(wpsList, dtype=float64)
    i = 0
    # while i + size < len(wpsArray):
    #     end = i + size
    #     if i + size >= len(wpsArray):
    #         end = len(wpsArray)
    #     med = np.median(wpsArray[i:end])
    #     print('med = ', med)
    #     if med == 0:
    #         med = 1
    #     medArr = np.array([med for j in range(end - i)], dtype=float64)
    #
    #     wpsArray[i:end] = np.subtract(wpsArray[i:end], medArr)
    #     wpsArray[i:end] = np.divide(wpsArray[i:end], medArr)
    #     i += size
    # wpsArray = preprocessing.StandardScaler().fit_transform(wpsArray)
    # wpsArray = preprocessing.scale(wpsArray)
    # wpsArray = preprocessing.minmax_scale(wpsArray)
    # mid = wpsArray.min()
    # midArr = np.array([-mid * 2 for i in range(len(wpsArray))])
    # wpsArray = np.add(wpsArray,midArr)
    # wpsArray = np.log(wpsArray)
    wpsArray = preprocessing.scale(wpsArray)
    print('normalized done')
    return wpsArray


###################Adjust wps
# find median and adjust to 0;
# and trim flank 120bp

def AdjustWPS(wpsList):
    n = len(wpsList)
    # lenth = n - win - 1
    subarray = wpsList[0: n]
    tmpLength = len(subarray)
    # medianA = np.zeros(tmpLength)
    # chaA = np.zeros(tmpLength)
    medianA = [0] * tmpLength
    chaA = [0] * tmpLength
    adjustWin = 1000
    mid = 500
    start = 0
    end = adjustWin + 1
    while end <= tmpLength:
        tmpArray = subarray[start: end]
        minn, median, maxn = getMMM(tmpArray)
        tmploc = int((start + end + 1) / 2)
        medianA[tmploc] = median
        chaA[tmploc] = maxn - minn + 1
        start += 1
        end += 1
    x = 0
    while x < tmpLength:
        loc = x
        if loc < 501:
            loc = 501
        if loc >= tmpLength - 501:
            loc = tmpLength - 501
        # print(chaA)
        subarray[x] = (subarray[x] - medianA[loc]) / chaA[loc]
        x += 1
    return np.array(subarray)

#@jit
def getMMM(tmpArray):
    tmpArray = np.array(tmpArray)
    return np.min(tmpArray), np.median(tmpArray), np.max(tmpArray)


def STL(x, wpsList, cycle):  # Seasonal-Trend decomposition procedure based on Loess 基于Loess（局部加权回归）的时间序列分解算法
    x = np.array(x).T
    wpsList = np.array(wpsList).T
    list = []
    list.append(x)
    list.append(wpsList)
    list = np.array(list).T
    array = np.array(list)
    df = pd.DataFrame(array)
    df.columns = ['pos', 'wps']
    df.set_index('pos')
    df['pws'] = df['wps'].apply(pd.to_numeric, errors='ignore')
    df.pws.interpolate(inplace=True)
    res = sm.tsa.seasonal_decompose(df.pws, freq=cycle)
    res.plot()
    plt.show()
    # resid seasonal trend
    print('STL done')
    return res.resid, res.seasonal


def lowess(wpsList):
    wpsList = np.array(wpsList)
    x = np.array([i for i in range(len(wpsList))])

    # f_hat = lo.lowess(x, wpsList, x)
    # plt.plot(x, wpsList, 'b')
    plt.plot(x, wpsList, 'b')
    plt.show()


# def waveletSmooth(wpsList, threshold):
#     w = pywt.Wavelet('db8')  # 选用Daubechies8小波
#     maxlev = pywt.dwt_max_level(len(wpsList), w.dec_len)
#     # print("maximum level is " + str(maxlev))
#     # Threshold for filtering
#
#     # Decompose into wavelet components, to the level selected:
#     coeffs = pywt.wavedec(wpsList, 'db8', level=maxlev)  # 将信号进行小波分解
#
#     # plt.figure()
#     for i in range(1, len(coeffs)):
#         coeffs[i] = pywt.threshold(coeffs[i], threshold * max(coeffs[i]))  # 将噪声滤波
#     datarec = pywt.waverec(coeffs, 'db8')  # 将信号进行小波重构

    # smoothData = smooth(datarec, 31)
    # smoothData = smooth(smoothData, 9)

    # fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10,6), dpi = 300)
    #
    # line1 = axes[0].plot(np.array([i for i in range(len(datarec))]), datarec, 'r')
    # line2 = axes[1].plot(np.array([i for i in range(len(smoothData))]), smoothData, 'k')
    # line3 = axes[2].plot(np.array([i for i in range(len(wpsList))]), wpsList, 'b')
    # plt.legend((line1[0], line2[0], line3[0]), ('小波滤波数据', '平滑数据', '原始数据'), loc='upper right')
    # plt.show()
    # return datarec


# def kalmanSmooth(wpsList):
#     wpsList = np.array(wpsList)
#     x = np.array([i for i in range(len(wpsList))])
#     measurements = np.asarray([[x[i], wpsList[i]] for i in range(len(wpsList))])
#     kf = KalmanFilter(transition_matrices=[[1, 1], [0, 1]], observation_matrices=[[0.1, 0.5], [-0.3, 0.0]])
#     # kf = kf.em(measurements, n_iter=5)
#     (smoothed_state_means, smoothed_state_covariances) = kf.smooth(measurements)
#     # draw estimates
#     pl.figure()
#     lines_true = pl.plot(measurements, color='b')
#     lines_smooth = pl.plot(smoothed_state_means, color='g')
#     pl.legend((lines_true[0], lines_smooth[0]),
#               ('true', 'smooth'),
#               loc='lower right'
#               )
#     pl.show()

#@jit
def kalman_filter(kalman_filter, wpsList):
    """
    :param wpsList: 待滤波数据
    :param Q: 过程噪声
    :param R: 测量噪声
    :return: 滤波后的数据
    """
    adc_filter_1 = np.zeros(len(wpsList))
    for i in range(len(wpsList)):
        adc_filter_1[i] = kalman_filter.kalman(wpsList[i])
    return adc_filter_1

def medfilt(wpsList, filterWin):
    wpsList = scipy.signal.medfilt(wpsList, filterWin)
    return wpsList


#@jit
def smooth(wpsList, WSZ):
    '''
    波形平滑
    :param a: 原始数据，NumPy 1-D array containing the data to be smoothed
               必须是1-D的，如果不是，请使用 np.ravel()或者np.squeeze()转化
    :param WSZ: smoothing window size needs, which must be odd number
    :return:
    '''
    out0 = scipy.signal.fftconvolve(wpsList, np.ones(WSZ, dtype=int), 'valid') / WSZ  # 对原始数据对于窗口长度为WSZ的进行卷积
    # np.ones(WSZ, dtype=int) 生成长度为WSZ的一维数组
    r = np.arange(1, WSZ - 1, 2)
    start = np.cumsum(wpsList[:WSZ - 1])[::2] / r
    stop = (np.cumsum(wpsList[:-WSZ:-1])[::2] / r)[::-1]
    smoothData = np.concatenate((start, out0, stop))  # 数组拼接
    # x = [i for i in range(len(wpsList))]
    # x = np.array(x)  # list to ndarray
    # start = 0
    # end = len(wpsList) - 1
    # plt.plot(x[start:end],wpsList[start:end],'b')
    # plt.plot(x[start:end], smoothData[start:end], 'r')
    # plt.show()
    print('smooth done')
    return smoothData

# #@jit
def getALLPeaksAveHeight(peakObjectList, normaliedRawDataList):
    for i in range(len(peakObjectList)):
        peak = peakObjectList[i]
        if peak.endPos != peak.startPos:
            aveHeight = getPeakAveHeight(normaliedRawDataList, peak.startPos, peak.endPos) / (
                    peak.endPos - peak.startPos)
            peak.aveHeight = aveHeight
        else:
            peak.aveHeight = 1
        # if i < len(peakObjectList) - 1 and abs(peak.endPos - peakObjectList[i + 1].startPos) > 150:
        #     nonePeakAveHeight = getPeakAveHeight(normaliedRawDataList, peak.endPos, peakObjectList[i + 1].startPos) / (peakObjectList[i + 1].startPos - peak.endPos)
        #     if nonePeakAveHeight != 0:
        #         nonePeakAreaList.append(nonePeakAveHeight)
    # print('none peak area ---------------------------------------- ', getPeakAveHeight(normaliedRawDataList, peak.endPos, peakObjectList[i + 1].startPos) / (peakObjectList[i + 1].startPos - peak.endPos))


def fft(wpsList, rawDataList, flag):
    ''''''
    fiterSize = len(wpsList)
    nyq = fiterSize * 0.5
    xft = np.linspace(0, len(wpsList), fiterSize)
    freqs1, yf1 = callNDR.fft(wpsList, fiterSize)
    # b, a = signal.iirdesign(1000 / 4000.0, 1100 / 4000.0, 1, 10, 0, "cheby1")
    b, a = scipy.signal.butter(4, 50 / nyq, "lowpass")  # lowpass
    # 阶数；最大纹波允许低于通频带中的单位增益。以分贝表示，以正数表示；频率(Hz)/奈奎斯特频率（采样率*0.5）
    b, a = scipy.signal.cheby1(4, 5, 50 / nyq, "lowpass")  # lowpass
    adjustWpsList = scipy.signal.filtfilt(b, a, wpsList)
    # xf = average_fft(y, 20000)
    freqs2, yf2 = callNDR.fft(adjustWpsList, fiterSize)
    if flag:
        fig, axes = plt.subplots(2, 1, figsize=(10, 4))
        axes[0].grid(linestyle="--")
        axes[1].grid(linestyle="--")
        line1 = axes[0].plot(freqs2[0:400], yf2[0:400], 'r')
        line2 = axes[1].plot(freqs1[0:400], yf1[0:400], 'b')
        plt.legend((line1[0], line2[0]), ('对原始数据进行低通滤波后，经快速傅里叶变换得到的频谱', '原始数据经傅里叶变换得到的频谱'), loc='lower right')
        plt.xlabel('频率（HZ）')
        plt.show()
        step2 = 2500
        for i in range(0, len(adjustWpsList), step2):
            fig, axes = plt.subplots(2, 1, figsize=(10, 4))
            axes[0].grid(linestyle="--")
            axes[1].grid(linestyle="--")
            lineData1 = axes[0].plot(xft[i:i + step2], adjustWpsList[i:i + step2], 'r')
            lineData2 = axes[1].plot(xft[i:i + step2], rawDataList[i:i + step2], 'b')
            plt.legend((lineData1[0], lineData2[0]), ('对原始数据进行低通滤波后的数据', '原始数据'), loc='lower right')
            plt.xlabel('基因位点')
            axes[0].set_ylabel(('低通滤波后的标准化wps值'))
            axes[1].set_ylabel('原始wps值')
            plt.show()
    return freqs2, yf2, freqs1, yf1, adjustWpsList

def myfft(wpsList, flag):
    # 采样点选择2048个，因为设置的信号频率分量最高为600赫兹，根据采样定理知采样频率要大于信号频率2倍，所以这里设置采样频率为1400赫兹（即一秒内有1400个采样点，一样意思的）
    fftSize = min(4096, len(wpsList))
    x = np.arange(0, fftSize, 1)
    wpsList = wpsList[0:fftSize]
    # 设置需要采样的信号，频率分量有200，400和600

    fft_y = np.fft.fft(wpsList, fftSize)  # 快速傅里叶变换

    N = fftSize
    x = np.arange(N)  # 频率个数
    half_x = x[range(int(N / 2))]  # 取一半区间

    abs_y = np.abs(fft_y)  # 取复数的绝对值，即复数的模(双边频谱)
    angle_y = np.angle(fft_y)  # 取复数的角度
    normalization_y = abs_y / N  # 归一化处理（双边频谱）
    normalization_half_y = normalization_y[range(int(N / 2))]  # 由于对称性，只取一半区间（单边频谱）

    if flag:
        plt.figure(figsize=(10,6))
        plt.subplot(211)
        plt.plot(x, wpsList, 'r')
        plt.title('原始波形', fontsize=9, color='k')

        # plt.subplot(232)
        # plt.plot(x, fft_y, 'black')
        # plt.title('双边振幅谱(未求振幅绝对值)', fontsize=9, color='black')
        #
        # plt.subplot(233)
        # plt.plot(x, abs_y, 'r')
        # plt.title('双边振幅谱(未归一化)', fontsize=9, color='red')
        #
        # plt.subplot(234)
        # plt.plot(x, angle_y, 'violet')
        # plt.title('双边相位谱(未归一化)', fontsize=9, color='violet')
        #
        # plt.subplot(235)
        # plt.plot(x, normalization_y, 'g')
        # plt.title('双边振幅谱(归一化)', fontsize=9, color='green')

        plt.subplot(212)
        plt.plot(half_x[0:400], normalization_half_y[0:400], 'blue')
        plt.title('单边振幅谱(归一化)', fontsize=9, color='blue')
        plt.show()

        # b, a = scipy.signal.butter(4, 75 / int(N/4), "lowpass")  # lowpass
        # # 阶数；最大纹波允许低于通频带中的单位增益。以分贝表示，以正数表示；频率(Hz)/奈奎斯特频率（采样率*0.5）
        # b, a = scipy.signal.cheby1(4, 5, 75 / int(N/4), "lowpass")  # lowpass
        # fWpsList = scipy.signal.filtfilt(b, a, wpsList)
        # normalization_half_fy = (np.abs(np.fft.fft(fWpsList)) / N)[range(int(N / 2))]
        # plt.figure(figsize=(10, 6))
        # plt.subplot(211)
        # plt.plot(x, fWpsList, 'r')
        # plt.title('低通滤波波形', fontsize=9, color='k')
        # plt.subplot(212)
        # plt.plot(half_x[0:400], normalization_half_fy[0:400], 'blue')
        # plt.title('低通过滤后单边振幅谱(归一化)', fontsize=9, color='blue')
        # plt.show()
    return normalization_half_y

#@jit
def fastFilter(wpsList, peaks, flag):
    # wpsSum = np.zeros(len(wpsList) + 1)
    # for i in range(len(wpsList)):
    #     wpsSum[i + 1] = wpsSum[i] + wpsList[i]

    peaksX = peaks[1]
    peakObjectList = peaks[2]
    squareWave = np.zeros(len(wpsList))
    if len(peakObjectList) <= 1:
        return 1000, 1000, 1000
    # leftKList = []
    # rightKList = []
    # aveHeight = []
    aveWidth = []
    aveDis = []
    aveArea = []
    lowLength = 0
    varWidth = 0
    lowCnt = 0
    for i in range(len(peakObjectList)):
        peak = peakObjectList[i]
        squareWave[peak.startPos:peak.endPos] = wpsList[peak.peakIndex] - min(wpsList[peak.startPos], wpsList[peak.endPos])
        if wpsList[peak.peakIndex] > 0.2:
            lowLength += peak.endPos - peak.startPos
            aveArea.append(peak.width * (peak.endPos - peak.startPos))
            lowCnt += 1
        # leftKList.append(peak.leftK)
        # rightKList.append(peak.rightK)
        # aveHeight.append((wpsSum[peak.endPos] - wpsSum[peak.startPos]) / peak.width)
        aveWidth.append(peak.width)
    # aveHeight = np.array(aveHeight)
    aveWidth = np.array(aveWidth)
    if lowCnt == 0:
        return 1000
    else:
        lowLength = (len(wpsList) - lowLength) / lowCnt
    varWidth =  np.var(aveWidth)
    # dis = diff(peaks[1])
    # bigDis = np.where(dis > 500)
    # print('dis : ', dis, ' len bigDis : ', len(bigDis[0]), ' bigDIs : ', bigDis)
    if flag:
        fig, axes = plt.subplots(2, 1, figsize=(12, 6))
        axes[0].grid(linestyle='--')
        axes[1].grid(linestyle='--')
        line1 = axes[0].plot(wpsList, 'r')
        line2 = axes[1].plot(squareWave, 'b')
        title('原始信号简单过滤为方波信号,其低于阈值的平均宽度 : ' + str(lowLength) + ' 宽度方差：' + str(varWidth))
        plt.xlabel('基因位点')
        axes[0].set_ylabel('wps值')
        axes[1].set_ylabel('wps值')
        plt.legend((line1[0], line2[0]), ('原始信号', '方波信号'), loc='lower right')
        plt.show()
    # leftKList = np.array(leftKList)
    # rightKList = np.array(rightKList)
    # print('leftKVar : ', np.var(leftKList), ' data : ', leftKList)
    # print('rightKVar : ', np.var(rightKList), ' data : ', rightKList)
    # print('aveAreaVar : ', np.var(aveArea), ' aveArea : ', aveArea)
    return lowLength, varWidth, squareWave

#@jit
def fastFilter2(wpsList):
    squareWave = np.zeros(len(wpsList))
    squareTemp = []
    cnt = 0
    for i in range(len(wpsList)):
        if wpsList[i] > -0.05:
            if cnt >= 20 and i >= 20:
                squareWave[i - cnt : i] = 0
            cnt = 0
            squareWave[i] = 1
        else:
            cnt += 1

    plt.grid(linestyle='--')
    plt.subplot(211)
    plt.plot(wpsList)
    plt.subplot(212)
    plt.plot(squareWave)
    plt.show()










def getPeakAveHeight(wpsList, startPos, endPos):
    '''
    :param wpsList: 原始数据
    :param startPos: 起始位置
    :param endPos: 终止位置
    :return:
    '''
    if endPos > len(wpsList) or startPos == endPos:
        return 0
    # print([startPos, endPos])
    x = np.arange(startPos, endPos, 1)
    minNum = np.min(wpsList[startPos:endPos])
    minLine = np.array([minNum for i in range(endPos - startPos)])
    # lowLine = np.array([-0.5 for i in range(len(x))])
    return scipy.integrate.simps(np.subtract(np.array(wpsList)[startPos:endPos], minLine), x=x)

#@jit
def scipy_signal_find_peaks(wpsFilterData, height, distance, prominence, width):

    peaksX, properties = scipy.signal.find_peaks(wpsFilterData, height=height, distance=distance, prominence=prominence,
                                                 width=width)
    # threshold : 和相邻峰垂直高度的阈值 None; min; [min, max]
    # prominence : 要求突出的峰的高度 None; minl; [min, max]
    # peaksX, properties = scipy.signal.find_peaks(wpsFilterData, height=0.025)
    # print("pearsX : ", peaksX)
    # print(properties)
    print('scipy_signal_find_peaks done')
    # properties
    return [wpsFilterData, peaksX]


def scipy_signal_find_peaks_cwt(wpsFilterData):
    wpsFilterData = np.array(wpsFilterData)
    peaksX = scipy.signal.find_peaks_cwt(wpsFilterData, np.arange(1, 61), max_distances=np.arange(1, 61) * 2,
                                         noise_perc=10)
    # vector, widths, wavelet = None, max_distances = None,
    # gap_thresh = None, min_length = None, min_snr = 1, noise_perc = 10
    # x = [i for i in range(win)]
    # plt.plot(x,wpsFilterData[0:x],'r')
    # plt.plot(peaksX, wpsFilterData[peaksX], 'xk')
    # plt.plot(peaksX, wpsFilterData[peaksX], 'xk')
    # plt.show()
    # print('scipy_signal_find_peaks_cwt done')
    print('scipy_signal_find_peaks_cwt done')
    return [wpsFilterData, peaksX]

#@jit
def getValley(wpsList, rawDataList, peaks, slidWinSize):
    leftValleyList = []
    rightValleyList = []
    for peakX in peaks:
        left = track2leftValley(wpsList, peakX, slidWinSize)
        right = track2rightValley(wpsList, peakX, slidWinSize)
        leftValleyList.append(left)
        rightValleyList.append(right)
    # x = np.array([i for i in range(len(wpsList))])
    # fig, axes = plt.subplots(2, 1)
    # axes[0].plot(x, wpsList, 'r')
    # axes[1].plot(x, rawDataList, 'b')
    # axes[0].vlines(leftValleyList,
    #                ymin=np.array(wpsList)[leftValleyList] - abs(np.array(wpsList)[leftValleyList] * 0.5),
    #                ymax=np.array(wpsList)[leftValleyList] + abs(np.array(wpsList)[leftValleyList] * 0.5), color='k')
    # axes[0].vlines(rightValleyList,
    #                ymin=np.array(wpsList)[rightValleyList] - abs(np.array(wpsList)[rightValleyList] * 0.5),
    #                ymax=np.array(wpsList)[rightValleyList] + abs(np.array(wpsList)[rightValleyList] * 0.5), color='g')
    # axes[1].vlines(leftValleyList,
    #                ymin=np.array(rawDataList)[leftValleyList] - abs(np.array(rawDataList)[leftValleyList] * 0.5),
    #                ymax=np.array(rawDataList)[leftValleyList] + abs(np.array(rawDataList)[leftValleyList] * 0.5),
    #                color='k')
    # axes[1].vlines(rightValleyList,
    #                ymin=np.array(rawDataList)[rightValleyList] - abs(np.array(rawDataList)[rightValleyList] * 0.5),
    #                ymax=np.array(rawDataList)[rightValleyList] + abs(np.array(rawDataList)[rightValleyList] * 0.5),
    #                color='g')
    # plt.show()
    # print(leftValleyList)
    # print(rightValleyList)
    peakObjectList = []

    for i in range(len(peaks)):
        leftPeakK = 0
        rightPeakK = 0
        # if rightValleyList[i] - leftValleyList[i] > 250:
        if peaks[i] - leftValleyList[i] != 0 and rawDataList[peaks[i]] != rawDataList[leftValleyList[i]]:
            leftPeakK = (rawDataList[peaks[i]] - rawDataList[leftValleyList[i]]) / (peaks[i] - leftValleyList[i])
        if peaks[i] - rightValleyList[i] != 0 and rawDataList[peaks[i]] != rawDataList[rightValleyList[i]]:
            rightPeakK = (rawDataList[peaks[i]] - rawDataList[rightValleyList[i]]) / (peaks[i] - rightValleyList[i])
        peakObjectList.append(
            Peak(peaks[i], leftValleyList[i], rightValleyList[i], rightValleyList[i] - leftValleyList[i], leftPeakK, rightPeakK))
    return peakObjectList

#@jit
def track2leftValley(wpsList, peakIndex, slidWinSize):
    win1 = wpsList[peakIndex - slidWinSize:peakIndex]
    win2 = wpsList[peakIndex - slidWinSize * 2:peakIndex - slidWinSize]
    win3 = wpsList[peakIndex - slidWinSize * 3:peakIndex - slidWinSize * 2]
    i = 3
    length = len(win1) + len(win2) + len(win3)
    x = np.array([i for i in range(0, length)])[:, np.newaxis]
    while (peakIndex - slidWinSize * i > 0 and slidWinSize * i <= 250 and not (
            np.sum(win1) > np.sum(win2) and np.sum(win2) < np.sum(win3))) or slidWinSize * i < 60:
        win1 = win2
        win2 = win3
        win3 = wpsList[peakIndex - slidWinSize * (i + 1): peakIndex - slidWinSize * i]

        i += 1
    if peakIndex - slidWinSize * i < 0 or slidWinSize * i > 250:
        if peakIndex > 80:
            return peakIndex - 80
        else:
            return 0
    valIndex = peakIndex - slidWinSize * i + int((slidWinSize + 1) / 2)
    curIndex = valIndex + int((slidWinSize + 1) / 2)
    while curIndex < peakIndex - 40 and abs(wpsList[valIndex] - wpsList[curIndex]) < 0.05:
        curIndex += 1

    return curIndex
    # model = linear_model.LinearRegression()
    # y = np.concatenate((win1,win2,win3))[:, np.newaxis]
    # minlen = min(len(x), len(y))
    # model.fit(x[0:minlen], y[0:minlen])
    # print('倾斜程度为', abs(model.coef_) * 1000)
    # print('end')

#@jit
def track2rightValley(wpsList, peakIndex, slidWinSize):
    win1 = wpsList[peakIndex:peakIndex + slidWinSize]
    win2 = wpsList[peakIndex + slidWinSize:peakIndex + slidWinSize * 2]
    win3 = wpsList[peakIndex + slidWinSize * 2:peakIndex + slidWinSize * 3]
    i = 3
    while ((not (np.sum(win1) > np.sum(win2) and np.sum(win2) < np.sum(win3)) and peakIndex + slidWinSize * i < len(
            wpsList) - 1) and slidWinSize * i <= 250) or slidWinSize * i < 60:
        win1 = win2
        win2 = win3
        win3 = wpsList[peakIndex + slidWinSize * i: peakIndex + slidWinSize * (i + 1)]
        i += 1
    if peakIndex + slidWinSize * i > len(wpsList):
        return len(wpsList) - 1
    valIndex = peakIndex + slidWinSize * i - int((slidWinSize + 1) / 2)
    curIndex = valIndex - int((slidWinSize + 1) / 2)
    while curIndex > peakIndex + 60 and abs(wpsList[valIndex] - wpsList[curIndex]) < 0.05:
        curIndex -= 1

    return curIndex


# def trackRightPeak(wpsList, startPos, slidWinSize):

#@jit
def mergeValley(peakObjectList):
    for i in range(len(peakObjectList)):
        if i == len(peakObjectList) - 1:
            break;
        if peakObjectList[i].endPos > peakObjectList[i + 1].peakIndex:
            peakObjectList[i].endPos = int((peakObjectList[i].peakIndex + peakObjectList[i + 1].peakIndex) / 2)
        elif peakObjectList[i + 1].startPos < peakObjectList[i].endPos:
            peakObjectList[i + 1].startPos = peakObjectList[i].endPos
    return peakObjectList

#@jit
def mergePeaks(peaks):
    '''
    :param peaks: scipy_signal_find_peaks方法找到的波形
    :param peaksCWT: scipy_signal_find_peaks_cwt方法找到的波形
    :return:
    '''

    for i in range(len(peaks)):
        if peaks[i].endPos - peaks[i].startPos > 200:
            if peaks[i].peakIndex - peaks[i].startPos > peaks[i].endPos - peaks[i].peakIndex:
                peaks[i].startPos = 2 * peaks[i].peakIndex - peaks[i].endPos
                peaks[i].width = 2 * (peaks[i].endPos - peaks[i].peakIndex)
            else:
                peaks[i].endPos = 2 * peaks[i].peakIndex - peaks[i].startPos
                peaks[i].width = 2 * (peaks[i].peakIndex - peaks[i].startPos)
    # for i in range(len(peaksCWT)):
    #     if peaksCWT[i].endPos - peaksCWT[i].startPos > 200:
    #         if peaksCWT[i].peakIndex - peaksCWT[i].startPos > peaksCWT[i].endPos - peaksCWT[i].peakIndex:
    #             peaksCWT[i].startPos = peaksCWT[i].endPos - 200
    #             peaksCWT[i].width = 200
    #         else:
    #             peaksCWT[i].endPos = peaksCWT[i].startPos + 200
    #             peaksCWT[i].width = 200
    return peaks


def findNDRPre(s, x, contig, peaks, peakObjectList, smoothData, rawDataList, label, color, smoothMethod,
               peakDisThreshold):
    '''
    :param contig: 染色体
    :param peaks:   波峰信息
    :param wpsList: 平滑数据
    :param rawDataList: 原始数据
    :param label:
    :param color: 颜色
    :param smoothMethod: 平滑采用的方法
    :param peakDisThreshold: 波峰间距的阈值
    :return: ndrInformation
    '''
    peaksX = peaks[1]
    s_Offset = np.array([s for i in range(len(peaksX))])
    peaksCorrectionX = np.add(peaksX, s_Offset)
    ndrDict = {}
    disArray = diff(peaksX, 1)

    # fig, axes = plt.subplots(2, 1)
    # ax1 = axes[0]  # 子图1
    # ax2 = axes[1]  # 子图2
    # ax1.plot(x, smoothData, color=color, label=label)
    # ax1.plot(peaksCorrectionX, smoothData[peaksX], 'x' + color, label='顶点')
    # ax2.plot(x, rawDataList, 'gray', label='原始数据')

    ndrDict['chromosome'] = []
    ndrDict['startPos'] = []
    ndrDict['endPos'] = []
    ndrDict['rawData'] = []
    ndrDict['smoothData'] = []
    ndrDict['smoothMethod'] = []
    for i in range(len(disArray)):
        if (i - 2 >= 0 and i + 2 < len(disArray)) and disArray[i] > peakDisThreshold and disArray[
            i - 1] < peakDisThreshold - 30 and disArray[i + 1] < peakDisThreshold - 30:
            ndrDict['chromosome'].append(contig)
            ndrDict['startPos'].append(peaksCorrectionX[i] + 30)
            ndrDict['endPos'].append(peaksCorrectionX[i + 1] - 30)
            ndrDict['smoothMethod'].append(smoothMethod)
            ndrDict['rawData'].append(rawDataList[peaksX[i] + 30: peaksX[i + 1] - 30])
            ndrDict['smoothData'].append(smoothData[peaksX[i] + 30: peaksX[i + 1] - 30])

            # yminPos = min(smoothData[peaksX[i] + 30], smoothData[peaksX[i + 1] - 30])
            # ymaxPos = max(smoothData[peaksX[i] + 30], smoothData[peaksX[i + 1] - 30])
            # ax1.vlines(x=peaksCorrectionX[i] + 30, ymin=yminPos - 0.2, ymax=ymaxPos + 0.2, colors='k', linestyles='dashed')
            # ax1.vlines(x=peaksCorrectionX[i + 1] - 30, ymin=yminPos - 0.2, ymax=ymaxPos + 0.2, colors='k',
            #            linestyles='dashed')
            # ax2.vlines(x=peaksCorrectionX[i] + 30, ymin=yminPos - 0.2, ymax=ymaxPos + 0.2, colors='k', linestyles='dashed')
            # ax2.vlines(x=peaksCorrectionX[i + 1] - 30, ymin=yminPos - 0.2, ymax=ymaxPos + 0.2, colors='k',
            #            linestyles='dashed')
    # ax1.legend(loc='best')
    # ax2.legend(loc='best')
    # plt.show()
    ndrInformation = pd.DataFrame(ndrDict)
    # print(ndrDict)
    return ndrInformation

def get_two_float(f_str, n):
    f_str = str(f_str)      # f_str = '{}'.format(f_str) 也可以转换为字符串
    a, b, c = f_str.partition('.')
    c = (c+"0"*n)[:n]       # 如论传入的函数有几位小数，在字符串后面都添加n为小数0
    return ".".join([a, c])

def findNDR(x, start, contig, peakObjectList, smoothData, rawDataList, label, color, smoothMethod, peakDisThreshold):
    '''
    :param start: 起始位置
    :param x:
    :param contig: 染色体标号
    :param peakObjectList: 波峰列表
    :param smoothData: 平滑数据
    :param rawDataList: 原始数据
    :param label:
    :param color: 颜色
    :param smoothMethod: 平滑方法
    :param peakDisThreshold: 阈值
    :return:
    '''
    ndrDict = {}
    ndrObjectList = []
    # normalizedRawDataList = preprocessing.minmax_scale(rawDataList)
    # fig, axes = plt.subplots(2, 1)
    # ax1 = axes[0]  # 子图1
    # ax2 = axes[1]  # 子图2
    # ax1.plot(x, smoothData, color=color, label=label)
    # ax1.plot(peaksCorrectionX, smoothData[peaksX], 'x' + color, label='顶点')
    # ax2.plot(x, rawDataList, 'gray', label='原始数据')

    # ndrDict['chromosome'] = []
    # ndrDict['startPos'] = []
    # ndrDict['endPos'] = []
    # ndrDict['rawData'] = []
    # ndrDict['smoothData'] = []
    # ndrDict['smoothMethod'] = []
    for i in range(len(peakObjectList)):
        if i < len(peakObjectList) - 1 and (
                peakObjectList[i + 1].startPos - peakObjectList[i].endPos > 120 or peakObjectList[i + 1].peakIndex -
                peakObjectList[i].peakIndex > 300 and peakObjectList[i + 1].peakIndex -
                peakObjectList[i].peakIndex < 500):
            if peakObjectList[i + 1].startPos != peakObjectList[i].endPos:
                aveHeight = getPeakAveHeight(smoothData, peakObjectList[i].endPos,
                                             peakObjectList[i + 1].startPos) / (
                                    peakObjectList[i + 1].startPos - peakObjectList[i].endPos)
                if aveHeight != 0:
                    ndr = NDR(peakObjectList[i].endPos, peakObjectList[i + 1].startPos,
                              peakObjectList[i + 1].startPos - peakObjectList[i].endPos)
                    ndr.aveHeight = aveHeight
                    kRight, kLeft, kNDR, ndrArea, ndrMax = judgeNDR(smoothData, rawDataList, ndr, peakObjectList, contig, start, slidWinSize=6, flag=False)
                    coef, mse = linearJudgeNDR(smoothData, rawDataList, ndr, False)
                    # print([coef, mse])
                    varWidth, varHeight, varAngel, varDis, varArea = haveNearContinuouslyPeak(smoothData, rawDataList, peakObjectList, 5, ndr, False)
                    # if (kLeft < 0.2 and kRight < 0.2 and coef[0][0] * 1000 < 2 and mse < 5):
                    condition = [varWidth < 800, varHeight < 200, varAngel < 180, varDis < 500, varArea < 150]
                    condCnt = 0
                    if varDis >= 2000 or varHeight > 2000 or varAngel >= 500 or varWidth >= 2000 and varArea >= 800:
                        continue
                    for con in condition:
                        if con:
                            condCnt += 1
                    if condCnt < 3 and varDis >= 1000 and varAngel >= 100 and varWidth >= 200 and varArea >= 100:
                        continue
                    if (kLeft < 0.22 and kRight < 0.22 and kNDR < 0.25 and ndrArea < 5.5 and coef[0][0] * 1000 < 3 and mse < 4):
                        kLeft, kRight, kNDR, ndrArea, ndrMax = judgeNDR(smoothData, rawDataList,   ndr, peakObjectList, contig, start, slidWinSize=6,
                                                 flag=True)
                        midPoint = int((ndr.endPos + ndr.startPos) / 2)
                        ndr.startPos = midPoint - 85
                        ndr.endPos = midPoint + 85
                        ndrObjectList.append(ndr)

                        # myfft(rawDataList[midPoint - 1024 : midPoint + 1024])
                        print('find a NDR')
                        varWidth, varHeight, varAngel, varDis, varArea = haveNearContinuouslyPeak(smoothData, rawDataList, peakObjectList, 5, ndr, False)
                        # drawSingleNDR(smoothData, rawDataList, x, contig, start, ndr)
                        # print('kRight : ', kRight, '   kLeft : ', kLeft)
                        # ndrDict['chromosome'].append(contig)
                        # ndrDict['startPos'].append(start + peakObjectList[i].endPos)
                        # ndrDict['endPos'].append(start + peakObjectList[i + 1].startPos)
                        # ndrDict['smoothMethod'].append(smoothMethod)
                        # ndrDict['rawData'].append(rawDataList[peakObjectList[i].endPos:peakObjectList[i + 1].startPos])
                        # ndrDict['smoothData'].append(
                        #     smoothData[peakObjectList[i].endPos:peakObjectList[i + 1].startPos])

            # yminPos = min(smoothData[peaksX[i] + 30], smoothData[peaksX[i + 1] - 30])
            # ymaxPos = max(smoothData[peaksX[i] + 30], smoothData[peaksX[i + 1] - 30])
            # ax1.vlines(x=peaksCorrectionX[i] + 30, ymin=yminPos - 0.2, ymax=ymaxPos + 0.2, colors='k', linestyles='dashed')
            # ax1.vlines(x=peaksCorrectionX[i + 1] - 30, ymin=yminPos - 0.2, ymax=ymaxPos + 0.2, colors='k',
            #            linestyles='dashed')
            # ax2.vlines(x=peaksCorrectionX[i] + 30, ymin=yminPos - 0.2, ymax=ymaxPos + 0.2, colors='k', linestyles='dashed')
            # ax2.vlines(x=peaksCorrectionX[i + 1] - 30, ymin=yminPos - 0.2, ymax=ymaxPos + 0.2, colors='k',
            #            linestyles='dashed')
    # ax1.legend(loc='best')
    # ax2.legend(loc='best')
    # plt.show()
    # ndrInformation = pd.DataFrame(ndrDict)
    # print(ndrDict)
    writeNDRToFile(contig, start, ndrObjectList)
    return ndrObjectList

##@jit
def judgeNDR(smoothData, rawDataList, squareWave,ndr, peakObjectList, contig, start, slidWinSize, flag):
    peakLeftIndex = trackLeftPeak(smoothData, rawDataList, ndr, slidWinSize=slidWinSize)
    peakRightIndex = trackRightPeak(smoothData, rawDataList, ndr, slidWinSize=slidWinSize)
    if peakRightIndex != peakLeftIndex:
        ndrMax = np.max(smoothData[min(peakRightIndex, peakLeftIndex) : max(peakLeftIndex, peakRightIndex)])
        print('ndr sum : ',(ndrMax - min(smoothData[peakRightIndex], smoothData[peakLeftIndex])))
    else:
        print('ndr sum : ', 0)
    xPeak = np.array([ndr.startPos, peakRightIndex, peakLeftIndex, ndr.endPos])
    xPeak = sort(xPeak)
    if xPeak[1] - xPeak[0] == 0:
        kLeft = 0
    else:
        kLeft = abs(smoothData[xPeak[1]] - smoothData[xPeak[0]]) / abs(xPeak[1] - xPeak[0]) * 100
    if xPeak[3] - xPeak[2] == 0:
        kRight = 0
    else:
        kRight = abs(smoothData[xPeak[3]] - smoothData[xPeak[2]]) / abs(xPeak[3] - xPeak[2]) * 100
    kNDR = abs(smoothData[xPeak[3]] - smoothData[xPeak[0]]) / abs(xPeak[3] - xPeak[0]) * 100
    if smoothData[xPeak[1]] > smoothData[xPeak[2]]:
        topPeak = xPeak[1]
    else:
        topPeak = xPeak[2]
    ndrArea = getTriangleArea([float(xPeak[0]), smoothData[xPeak[0]]], [float(topPeak), smoothData[topPeak]],[float(xPeak[3]), smoothData[xPeak[3]]])
    if flag:
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(14, 4))
        xStrat = max(ndr.startPos - 2500, 0)
        xEnd = min(ndr.endPos + 2500, min(len(smoothData), len(rawDataList)))
        x = np.array([i + start for i in range(xStrat, xEnd)])
        line1 = axes[0].plot(x, smoothData[xStrat: xEnd], 'r')
        axes[0].plot(xPeak + start, np.array(smoothData)[xPeak], 'k')
        line2 = axes[1].plot(x, rawDataList[xStrat: xEnd], 'b')

        # ax1.vlines(x=peaksCorrectionX[i] + 30, ymin=yminPos - 0.2, ymax=ymaxPos + 0.2, colors='k', linestyles='dashed')

        axes[1].plot(xPeak + start, rawDataList[xPeak], 'k')
        axes[0].plot(xPeak + start, smoothData[xPeak], 'xk')
        axes[1].plot(xPeak + start, np.array(rawDataList)[xPeak], 'xk')
        axes[1].plot(np.array([xPeak[0], xPeak[3]]) + start, rawDataList[np.array([xPeak[0], xPeak[3]])], 'r')
        axes[0].plot(np.array([xPeak[0], xPeak[3]]) + start, smoothData[np.array([xPeak[0], xPeak[3]])], 'r')
        axes[0].set_title(
            'ndr.aveHeight : ' + str(get_two_float(ndr.aveHeight, 2)) + '  kLeft : ' + str(get_two_float(kLeft, 2)) + '  kRight : ' + str(get_two_float(kRight, 2))
        + ' kNDR : ' + str(get_two_float(kNDR, 2)) + ' ndrArea : ' + str(get_two_float(ndrArea, 2)))

        list = np.array([ndr.startPos, ndr.endPos])
        axes[0].vlines(x=list + start, ymin=smoothData[list] - 0.15,
                       ymax=smoothData[list] + 0.15, colors='k', linestyles='dashed')

        axes[1].vlines(x=list + start, ymin=rawDataList[list] - abs(rawDataList[list] * 1.4),
                       ymax=rawDataList[list] + abs(rawDataList[list] * 1.4), colors='k', linestyles='dashed')
        #
        plt.legend((line1[0], line2[0]), ('平滑后的WPS曲线', '原始WPS曲线'), loc='best')
        axes[1].set_title('疑似NDR区域')
        plt.xlabel(str(contig) + '号染色体基因组位点')
        axes[0].set_ylabel('标准化wps值')
        plt.ylabel('wps值')
        plt.show()
    print('judgeNDR done')
    return kLeft, kRight, kNDR, ndrArea, ndrMax

#@jit
def getPointDis(point0, point1):
    return ((point0[0] - point1[0]) ** 2 + (point0[1] - point1[1]) ** 2) ** 0.5

#@jit
def getTriangleArea(point0, point1, point2):
    line0 = getPointDis(point0, point1)
    line1 = getPointDis(point0, point2)
    line2 = getPointDis(point1, point2)
    p = (line0 + line1 + line2) / 2
    return (p*(p - line0)*(p - line1)*(p - line2))** 0.5

def haveNearContinuouslyPeak(smoothData, rawDataList, peakObjectList, cnt, ndr, flag):
    peakLeftList = []
    peakRightList = []
    for i in range(len(peakObjectList)):
        if peakObjectList[i].peakIndex < ndr.endPos:
            peakLeftList.append(peakObjectList[i])
            continue;
        if len(peakRightList) >= cnt:
            break;
        peakRightList.append(peakObjectList[i])
    if len(peakLeftList) >= cnt:
        peakLeftList = peakLeftList[-cnt:]
    nearPeakList = peakLeftList + peakRightList
    peakListSize = len(nearPeakList)
    peakWidth = np.zeros(peakListSize)
    peakHeight = np.zeros(peakListSize)
    peakAngel = np.zeros(peakListSize)
    peakDis = []
    peakArea = np.zeros(peakListSize)
    nearPeakX = []
    peakVell = []
    maxDisPeakCount = 0
    for i in range(peakListSize):
        peak = nearPeakList[i]
        nearPeakX.append(peak.peakIndex)
        peakVell.append(peak.startPos)
        peakVell.append(peak.endPos)
        peakWidth[i] = peak.width
        peakHeight[i] = rawDataList[peak.peakIndex] - min(rawDataList[peak.startPos], rawDataList[peak.endPos])
        if 1 + peak.leftK*peak.rightK != 0:
            peakAngel[i] = math.atan(abs((peak.leftK - peak.rightK) / (1 + peak.leftK*peak.rightK))) * 180 / 3.1415
        else:
            peakAngel[i] = 90
        peakArea[i] = getTriangleArea([float(peak.startPos), rawDataList[peak.startPos]],
                                      [float(peak.peakIndex), rawDataList[peak.peakIndex]],
                                      [float(peak.endPos), rawDataList[peak.endPos]]) / 100
        if i < peakListSize - 1 and nearPeakList[i + 1].peakIndex - peak.peakIndex < 300 and maxDisPeakCount < 5:
            peakDis.append(nearPeakList[i + 1].peakIndex - peak.peakIndex)
        elif i < peakListSize - 1 and nearPeakList[i + 1].peakIndex - peak.peakIndex > 450:
            maxDisPeakCount += 1;
        elif i < peakListSize - 1 and maxDisPeakCount >= 5:
            peakDis.append(10000)
    varDis = np.var(np.array(peakDis))
    varWidth = np.var(peakWidth)
    varHeight = np.var(peakHeight)
    varAngel = np.var(peakAngel)
    varArea = np.var(peakArea)
    nearPeakX = np.array(nearPeakX)
    peakVell = np.array(peakVell)
    if flag:
        fig, axes = plt.subplots(2, 1, figsize=(10, 5))
        axes[0].plot(np.array([i for i in range(nearPeakList[0].startPos, nearPeakList[-1].endPos)]),
                     smoothData[nearPeakList[0].startPos:nearPeakList[-1].endPos], 'r')
        axes[1].plot(np.array([i for i in range(nearPeakList[0].startPos, nearPeakList[-1].endPos)]),
                     rawDataList[nearPeakList[0].startPos:nearPeakList[-1].endPos], 'b')
        axes[0].set_title(
            'width var : ' + get_two_float(str(varWidth), 2) + '   height var : ' + get_two_float(str(varHeight),2)
            + '   angel var : ' + get_two_float(str(varAngel), 2) +
            '   varDis : ' + get_two_float(str(varDis), 2) + '  varArea : ' + get_two_float(str(varArea), 2))
        axes[0].plot(nearPeakX, smoothData[nearPeakX], 'xr')
        axes[1].plot(nearPeakX, rawDataList[nearPeakX], 'xb')
        axes[0].vlines(x = peakVell, ymin=smoothData[peakVell] - 0.2, ymax=smoothData[peakVell] + 0.2)
        axes[1].vlines(x = peakVell, ymin=rawDataList[peakVell] - 30, ymax=rawDataList[peakVell] + 30)
        plt.show()
        print('width var : ', varWidth, ' width : ', peakWidth)
        print('height var : ', varHeight, ' height : ', peakHeight)
        print('angel var : ', varAngel, ' angel : ', peakAngel)
        print('Dis var : ', varDis, ' Dis : ', peakDis)
    print('haveNearContinuouslyPeak done')
    return varWidth, varHeight, varAngel, varDis, varArea

def linearJudgeNDR(smoothData, rawDataList, ndr, flag):
    model = linear_model.LinearRegression()
    x = np.array([i for i in range(ndr.startPos, ndr.endPos)])[:, np.newaxis]
    y = smoothData[ndr.startPos:ndr.endPos][:, np.newaxis]
    model.fit(x, y)
    if flag:
        print(model.intercept_)  # 截距
        print(model.coef_)
        plt.plot(x, y, marker = '.', color = 'r')
        plt.plot(x, model.coef_*x + model.intercept_, color = 'k')
        title('直线为 ： ' + str(model.coef_) + 'x + ' + str(model.intercept_) + '    均方误差 : ' + str(np.mean((model.predict(x) - y) ** 2)))
        plt.show()
    return model.coef_, np.mean((model.predict(x) - y) ** 2) * 1000

#@jit
def trackRightPeak(smoothData, rawDataList, ndr, slidWinSize):
    win1 = smoothData[ndr.startPos:ndr.startPos + slidWinSize]
    win2 = smoothData[ndr.startPos + slidWinSize:ndr.startPos + slidWinSize * 2]
    win3 = smoothData[ndr.startPos + slidWinSize * 2:ndr.startPos + slidWinSize * 3]
    i = 3
    while (not (np.sum(win1) < np.sum(win2) and np.sum(win2) > np.sum(win3)) and ndr.startPos + slidWinSize * (
            i + 1) < len(smoothData) - 1 and ndr.startPos + slidWinSize * (i + 1) < ndr.endPos) or np.sum(win2) < 0:
        win1 = win2
        win2 = win3
        win3 = smoothData[ndr.startPos + slidWinSize * i: ndr.startPos + slidWinSize * (i + 1)]
        i += 1
    if ndr.startPos + slidWinSize * i - 3 > ndr.endPos:
        return min(ndr.endPos, len(smoothData))
    return ndr.startPos + slidWinSize * i - int((slidWinSize + 1) / 2)

#@jit
def trackLeftPeak(smoothData, rawDataList, ndr, slidWinSize):
    win1 = smoothData[ndr.endPos - slidWinSize:ndr.endPos]
    win2 = smoothData[ndr.endPos - slidWinSize * 2:ndr.endPos - slidWinSize]
    win3 = smoothData[ndr.endPos - slidWinSize * 3:ndr.endPos - slidWinSize * 2]
    i = 3
    while (ndr.endPos - slidWinSize * 3 > 0 and ndr.endPos - slidWinSize * 3 > ndr.startPos and
           not (np.sum(win1) < np.sum(win2) and np.sum(win2) > np.sum(win3))) or np.sum(win2) < 0:
        win1 = win2
        win2 = win3
        win3 = smoothData[ndr.endPos - slidWinSize * (i + 1): ndr.endPos - slidWinSize * i]
        i += 1
    if ndr.endPos - slidWinSize * i + int((slidWinSize + 1) / 2) < 0:
        return ndr.startPos
    return ndr.endPos - slidWinSize * i + int((slidWinSize + 1) / 2)


# def find_peak_triangleNum(filter_signal, peak_width):
#     '''
#     判断序列数据波形的三角形个数
#     :param filter_signal: 平滑后的波形
#     :param step:连续几个
#     :param peak_width:尖峰之间的宽距小于peak_width时划分为一个峰，频域数据一般定义在20；
#                        时域数据三角形识别一般定义在15，太大会滤掉双峰
#     :return:
#     '''
#     # 判断是否有凸起
#     length_data = len(filter_signal)
#     thre = 0.7 * np.percentile(filter_signal, 95)  # 设置阈值高度  95%是前400个点的20个波峰点
#     # 在整个区域内找极值
#     l = []
#     for i in range(1, length_data - 1):
#         if filter_signal[i - 1] < filter_signal[i] and filter_signal[i] > filter_signal[i + 1] and filter_signal[
#             i] > thre:
#             l.append(i)
#         elif filter_signal[i] == filter_signal[i - 1] and filter_signal[i] > thre:
#             l.append(i)  # 最高点前后可能有相等的情况
#     CC = len(l)  # 统计极值得个数
#     cou = 0
#     ll = l.copy()
#     for j in range(1, CC):
#         if l[j] - l[j - 1] < peak_width:  # 此判断用于将位于同一个峰内的极值点去掉
#             if l[j] > l[j - 1]:  # 同一个峰内的数据，将小的值替换成0
#                 ll[j - 1] = 0
#             else:
#                 ll[j] = 0
#             cou = cou + 1
#     rcou = CC - cou
#     ll = [i for i in ll if i > 0]  # 去掉0的值
#     peak_index = []
#     # 找到每个区间内波峰最大值
#     # 截断每个区间再求区间最大值的索引
#     for i in range(len(ll)):
#         if i == 0:
#             index_range = np.array(l)[np.array(l) <= ll[i]]
#         else:
#             index_range = np.array(l)[(np.array(l) <= ll[i]) & (np.array(l) > ll[i - 1])]
#         # 找到每个区间最大值得索引
#         peak_index.append(index_range[np.argmax(filter_signal[index_range], axis=0)])
#     return [rcou, peak_index]


# def triangle_flag_meiquan(q_flush, n, ratio,max_index):
#     '''
#     判断三角形的形状，首先每400个点进行一次过滤，过滤的规则是value-max(value)*0.1
#     :param q_flush:
#     :param n: 一条振动数据包含的数据点数
#     :param ratio: 默认0.2或者0.3
#     :return:
#     '''
#     circle_range = [0]
#     zhankongbi = []
#     left = []
#     right = []
#     # 生成int_num*step 的矩阵，头尾需要判断 ,其中q_flush是list类型。
#     for i in range(len(max_index)-1):
#         next_index = int((max_index[i+1]-max_index[i])/2)+max_index[i]
#         circle_range.append(next_index)
#         q = q_flush[circle_range[i]:circle_range[i+1]]
#         # 每圈除噪
#         newq = q-max(q)*ratio
#         newq[newq < 0] = 0
#         # 占空比计算三角形形状判断
#         zhankongbi.append((max(max(np.where(newq > 0))) - min(min(np.where(newq > 0)))) /len(newq))
#         max_index_now = np.argmax(newq)
#         left.append(max_index_now - min(min(np.where(newq > 0))))
#         right.append(max(max(np.where(newq>0))) - max_index_now)
# # 进行三角形识别，不考虑左边为0
#     right_zb = np.array(zhankongbi)[np.where(np.array(right) >= np.array(left))]  # 右边大于等于左边对应占空比
#     right_count = np.sum(right_zb <= 0.3)                    # 右边大于左边占空比小于0.3的个数
#     left_zb = np.array(zhankongbi)[np.where(np.array(left) > np.array(right))]  # 左边大于等于右边对应占空比
#     left_count = np.sum(left_zb <= 0.2)                      # 左边大于右边但是占空比小于0.2
#     if left_count+right_count >= len(zhankongbi)/2:
#         flag = 2  # 直角三角形
#     else:
#         flag = 1   # 等腰三角形
#     return [flag, zhankongbi]


def getTssPoint(tssFile):
    tssPointList = []
    while True:
        line = tssFile.readline()
        if not line:
            break
        tssPoint = line.split()
        tssPoint[0] = int(tssPoint[0])
        tssPoint[1] = int(tssPoint[1])
        tssPoint[2] = int(tssPoint[2])
        tssPointList.append(tssPoint)
    return tssPointList


def getVChipPoint(fileVChip):
    v109_chip = []
    while True:
        line = fileVChip.readline()
        if not line:
            break
        vIndexlist = line.split()
        vIndexlist[0] = int(vIndexlist[0])
        vIndexlist[1] = int(vIndexlist[1])
        vIndexlist[2] = int(vIndexlist[2])
        v109_chip.append(vIndexlist)
    return v109_chip


def drawPeaksLength(peaksLen):
    peakCnt = [0 for i in range(600)]
    for i in peaksLen:
        if i < 600:
            peakCnt[i] += 1
    x = np.array([i for i in range(600)])
    xList = []
    yList = []
    for i in range(600):
        if peakCnt[i] != 0:
            xList.append(i)
            yList.append(peakCnt[i])
    plt.plot(xList, yList, 'r')
    plt.show()


def KernelDensityEstimate(kernel, bandwidth, dataList, start, end, point, bin, maxYlim, minYlim, title):
    '''
    :param dataList: 源数据
    :param start: x轴起点
    :param end: x轴终点
    :param point: 划分间隔数
    :param bin: 直方图箱子数
    :return:
    '''
    dataList = np.array(dataList)
    X = dataList[:, np.newaxis]
    # print(len(X))
    X_plot = np.linspace(start, end, point)[:, np.newaxis]
    bins = np.linspace(start, end, bin)
    fig, ax = plt.subplots(2, 1, sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    ax[0].set_title(title)
    ax[0].grid(axis="y", linestyle='--')
    # 直方图 1 'Histogram'
    # ax[0, 0].hist(X[:, 0], bins=bins, fc='#AAAAFF', normed=True)
    # ax[0, 0].text(-3.5, 0.31, 'Histogram')
    # 直方图 2 'Histogram, bins shifted'
    ax[0].hist(X[:, 0], bins=bins, fc='#AAAAFF', normed=True)
    # ax[0].text(-3.5, maxYlim, 'Histogram, bins shifted')
    # # 核密度估计 1 'tophat KDE'
    # kde = KernelDensity(kernel='tophat', bandwidth= 1).fit(X)
    # log_dens = kde.score_samples(X_plot)
    # ax[1, 0].fill(X_plot[:, 0], np.exp(log_dens), fc='#AAAAFF')
    # ax[1, 0].text(-3.5, 0.31, 'Tophat Kernel Density')
    # 核密度估计 2 'Gaussian KDE'
    kde2 = KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(X)

    log_dens = kde2.score_samples(X_plot)
    ax[1].fill(X_plot[:, 0], np.exp(log_dens), fc='#AAFFFF')
    ax[1].grid(axis="y", linestyle='--')
    # ax[1].text(-3.5, maxYlim, 'Gaussian Kernel Density')
    # print(scipy.integrate.simps(x = np.linspace(start, end, point)[100:1000],y = np.array(np.exp(log_dens)[100:1000]), dx = (end - start) / point))

    for axi in ax.ravel():
        axi.plot(X[:, 0], np.zeros(X.shape[0]) - 0.04, '+k')
        axi.set_xlim(start, end)
        axi.set_ylim(minYlim, maxYlim)
    for axi in ax.ravel():
        axi.set_ylabel('Normalized Density')
    for axi in ax.ravel():
        axi.set_xlabel('x')
    plt.savefig('kernelEstimate.png')
    plt.show()


def getPointData(pointFilePath, cnt):
    pointList = []
    file = open(pointFilePath, 'r')
    dataList = []
    while True:
        line = file.readline()
        if line == None or cnt < 0:
            break
        cnt -= 1
        data =line[0:-1].split('\t')
        if len(data) == 1:
            break
        if not str.isdigit(data[0]):
            continue
        dataList.append(data)
    for data in dataList:
        pointList.append([str(data[0]), int(data[1]), int(data[2])])
    return pointList

def writeNDRToFile(contig, start, ndrObjectList):
    # with open('ndrInfo.txt', mode='a+') as f:
    #     for ndr in ndrObjectList:
    #         list = [contig, start + ndr.startPos, start + ndr.endPos, start]
    #         list.extend(rawDataList[max(ndr.startPos - 1000, 0) : min(ndr.endPos + 1000, len(rawDataList))])
    #         f.write(str(list)+ '\n')
    with open('ndr_DHSAndTSS.chr10_e.txt', mode='a+') as f:
        for ndr in ndrObjectList:
            list = str(contig) + "\t" + str(start + ndr.startPos) + "\t" + str(start + ndr.endPos)
            f.write(str(list)+ '\n')

# main():

# def run():


if __name__ == '__main__':
    # dataList = []
    # list = []
    #
    # peakDis = []
    # filePath = '/home/chenlb/WPSCalProject/00.panel_data/panel.bam'
    pointFilePath = '/mnt/X500/farmers/chenhx/02.project/hexiaoti/05.NDR_all/wholegenome.20k.filter.bin'

    panelDataPathList = [
        '/mnt/GenePlus001/prod/workspace/IFA20180917003/OncoD/output/180016849BCD_180016848FD/cancer/5_recal_bam/180016849BCD_180016848FD_cancer_sort_markdup_realign_recal_ori.bam',
        # '/mnt/GenePlus001/prod/workspace/IFA20180917003/OncoD/output/180016849BCD_180016848FD/normal/5_recal_bam/180016849BCD_180016848FD_normal_sort_markdup_realign_recal_ori.bam',
        # '/mnt/GenePlus001/prod/workspace/IFA20181113007/OncoD/output/180019006BCD_180019006FD/cancer/5_recal_bam/180019006BCD_180019006FD_cancer_sort_markdup_realign_recal_ori.bam',
        # '/mnt/GenePlus001/prod/workspace/IFA20181126004/OncoD/output/180022065BCD_180025721FD/cancer/5_recal_bam/180022065BCD_180025721FD_cancer_sort_markdup_realign_recal_ori.bam'
    ]
    # filePath = '/mnt/GenePlus001/prod/workspace/IFA20180917003/OncoD/output/180016849BCD_180016848FD/cancer/5_recal_bam/180016849BCD_180016848FD_cancer_sort_markdup_realign_recal_ori.bam'
    sWGSDataPathList = [
        # 前三个为小细胞肺癌，临床分期IV
        # '/mnt/X500/farmers/chenhx/02.project/hexiaoti/02.cellPaper_data/IH01.bam',
        # '/mnt/X500/farmers/chenhx/02.project/hexiaoti/02.cellPaper_data/BH01.bam',
        # '/mnt/X500/farmers/chenhx/02.project/hexiaoti/02.cellPaper_data/IH02.bam',
        '/mnt/X500/farmers/chenhx/02.project/hexiaoti/01.sWGS_data/Ovarian.all.bam'
        # '/mnt/X500/farmers/chenhx/02.project/hexiaoti/01.sWGS_data/Healthy.20190108.bam',
        # '/mnt/X500/farmers/chenhx/02.project/hexiaoti/01.sWGS_data/Healthy.20190211.bam' 正常个体
        # '/mnt/X500/farmers/limin/ET/20190108/ET/output/bams/180016513BPD_markdup_realign_recald.bam',
        # '/mnt/X500/farmers/limin/ET/20190108/ET/output/bams/180016501BPD_markdup_realign_recald.bam',
        # '/mnt/X500/farmers/limin/ET/20190108/ET/output/bams/180016501BPD_markdup_realign_recald.bam'
        # '/mnt/X500/farmers/limin/zaoshai/ET/20190211/ET/output/bams/189006244BPD_markdup_realign_recald.bam',
        # '/mnt/X500/farmers/limin/zaoshai/ET/20190211/ET/output/bams/189006948BPD_markdup_realign_recald.bam'
        # '/mnt/X500/farmers/limin/zaoshai/ET/20190211/ET/output/bams/189006244BPD_markdup_realign_recald.bam'
    ]

    # fileVChipPath = '/home/chenlb/WPSCalProject/00.panel_data/v109_chip_more500.txt'
    # fileVChip = open(fileVChipPath, 'r')
    # v109_chip = getVChipPoint(fileVChip)

    # tssPath = '/home/chenlb/WPSCalProject/00.panel_data/TSS.txt'
    # tssFile = open(tssPath, 'r')
    # tssPointList = getTssPoint(tssFile)
    #
    # pointList = tssPointList
    pathList = sWGSDataPathList

    s = e = 0
    # for point in pointList:
    # areaList = []
    # nonePeakAreaList = []
    # dict = {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "10": 10, "11": 11, "12": 12,
    #         "13": 13, "14": 14, "15": 15, "16": 16, "17": 17, "18": 18, "19": 19, "20": 20, "21": 21, "22": 22, "X": 23,
    #         "Y": 24}
    # step = 4000
    # 3713568 3713568
    kal_filter = Kalman_filter.kalman_filter(Q=0.1, R=10)
    # peakWdith = []
    # contig = str(2)
    pointList = getPointData(pointFilePath, 1000000000)
    allPoint = len(pointList)
    # print(pointList)
    # for index in range(100000, 100000 + 400000, step):
    round = 0
    t1 = time.clock()
    for point in pointList:
        print('*************************************  round : ', round,'  ->  ', allPoint,' *************************************')
        round += 1
        if round > 20:
            break
        contig = str(point[0])
        start = point[1]
        end = point[2] - 16500
        step = end - start + 200
        peaksList = []
        dataList = []
        wpsList = np.array([0 for i in range(step + 1)])
        length = step + 1
        # wpsSegTree = SegmentTree(len(wpsList), wpsList)

        for filePath in pathList:
            start = point[1]
            end = point[2]
            bamfile = readFileData(filePath)
            win = 120
            # wpsList = wpsCal(bamfile, win, contig, start, end, s, e)
            # bamfile = readFileData(filePath)

            wpsList = wpsCal(wpsList, bamfile, win, contig, start, end, s, e)
            # wpsList, up, down = wpsCalBySegTree(wpsSegTree, wpsList, bamfile, win, contig, start, end, s, e)
        wpsList = np.array(wpsList, dtype=float)
            # ocf = callOCF(up, down)
        rawDataList = np.array(wpsList);

        adjustWpsList = AdjustWPS(wpsList)

        peaks = scipy_signal_find_peaks(adjustWpsList, height=0.01, distance=30, prominence=0.1, width=[30, 170])
        peakObjectList = getValley(adjustWpsList, rawDataList, np.array(peaks[1]), 4)
        peaks.append(peakObjectList)
        # fastFilter2(adjustWpsList)
        lowLength, varWidth, squareWave = fastFilter(adjustWpsList, peaks, True)
        if lowLength > 140 or varWidth > 250:
            print('lowLength : ', lowLength, ' varWidth : ', varWidth)
            continue
        # lowLength = fastFilter(adjustWpsList, peaks, True)






        # adjustWpsList = preprocessing.scale(wpsList)
        # adjustWpsList = waveletSmooth(adjustWpsList, threshold = 0.15)
        adjustWpsList = kalman_filter(kal_filter, adjustWpsList)
        adjustWpsList = smooth(adjustWpsList, 31)
        # adjustWpsList = smooth(adjustWpsList, 9)
        # ocf = AdjustWPS(ocf)
        # ocf = smooth(ocf, 51)
        # ocf = smooth(ocf, 51)

        # fig, axes = plt.subplots(2, 1)
        # lineWPS = plt.plot([i for i in range(len(adjustWpsList))], adjustWpsList,'r')
        # lineOCF = plt.plot([i for i in range(len(adjustWpsList))], ocf, 'k')
        # plt.legend((lineWPS[0], lineOCF[0]),('WPS', 'OCF'),loc='upper right')
        # plt.show()

        # base = peakutils.baseline(adjustWpsList, 2)

        # wpsList = normalized(wpsList, 500)
        # wpsList = AdjustWPS(wpsList)
        # rawDataList = wpsList

        # lowess(wpsList)
        # list.append(np.array(wpsList))

        # dataList.append(wpsList)
        # draw(dataList)
        # dataList = []

        # savgolFilterData = savgol_filter_func(wpsList)
        # smoothData = smooth(adjustWpsList, 51)
        # kalmanSmooth(smoothData)
        x = np.array([start + i for i in range(len(wpsList))])
        # smoothData = smooth(smoothData, 31)
        # [169 157 178 204 373 170  97 288 145 362 193 187 191 209 196 203 170 181
        #  212 183]
        # residualDatam, seasonal = STL(x, rawDataList, cycle = 312)

        # peaks = scipy_signal_find_peaks(adjustWpsList, height=0.05, distance=40, prominence=0.1, width=[30, 170])
        # peaks2 = scipy_signal_find_peaks(ocf, height=0.01, distance=30, prominence=0.1, width=[30, 180])
        # peaksCWT = scipy_signal_find_peaks_cwt(adjustWpsList)
        # peaks2 = scipy_signal_find_peaks_cwt(ocf)
        peaks = scipy_signal_find_peaks(adjustWpsList, height=0.03, distance=25, prominence=0.1, width=[25, 170])
        peakObjectList = getValley(adjustWpsList, rawDataList, peaks[1], 4)
        # peakObjectListCWT = getValley(adjustWpsList, rawDataList, peaksCWT[1], 4)
        #
        mergeValley(peakObjectList)
        # mergeValley(peakObjectListCWT)
        peakObjectList = mergePeaks(peakObjectList)

        # normaliedRawDataList = preprocessing.minmax_scale(rawDataList)

        # getALLPeaksAveHeight(peakObjectList, normaliedRawDataList=)

        # print(areaList)
        # KernelDensityEstimate(kernel='gaussian', bandwidth=0.02, dataList=areaList, start=0.35, end=0.65, point=100000,
        #                       bin=18, maxYlim=14, minYlim = -0.02)
        # KernelDensityEstimate(kernel='gaussian', bandwidth=0.02, dataList=nonePeakAreaList, start=0.35, end=0.65,
        #                       point=100000, bin=18, maxYlim=14, minYlim=-0.02)
        #                 # rightValleyList[i] = leftValleyList[i+1] = (rightValleyList[i] + leftValleyList[i+1]) / 2

        # for i in range(len(peakObjectList)):
        #     width = peakObjectList[i].endPos - peakObjectList[i].startPos
        #     if width > 80 and width < 230:
        #         peakWdith.append(width)
        # print('peakWidth : ', peakWdith)

        # SEQ = </mnt/X500/farmers/chenlb/SeqLib>
        # drawPeaksLength(np.array(peadWidth))
        # peaks.append(peakObjectList)

        # peaksCWT.append(peakObjectListCWT)
        # rawPeak = [rawDataList, np.array(peaks[1]), peakObjectList]
        # for peakX in peaks2[1]:
        #     peak = Peak(peakX - 80, peakX + 80, 160)
        #     getPeakProperties(peak, smoothData)  
        # print(diff(peaks[1]))
        # peaks3 = scipy_signal_find_peaks(rawDataList)
        # {263: 225, 488: 174, 662: 629, 1291: 403, 1694: 206, 1900: 205, 2105: 202, 2307: 256, 2563: 287, 2850: 172, 3022: 192, 3214: 120, 3334: 97, 3431: 99, 3530: 168, 3698: 186, 3884: 98, 3982: 159, 4141: 194, 4335: 141, 4476: 340, 4816: 185, 5001: 104, 5105: 470, 5575: 201, 5776: 480, 6256: 246, 6502: 196, 6698: 224, 6922: 134, 7056: 208, 7264: 192, 7456: 179, 7635: 180, 7815: 154, 7969: 222, 8191: 182, 8373: 150, 8523: 327, 8850: 304, 9154: 269, 9423: 174, 9597: 167, 9764: 159} c = 300
        # {261: 108, 369: 118, 487: 1056, 1543: 150, 1693: 206, 1899: 407, 2306: 713, 3019: 194, 3213: 119, 3332: 100, 3432: 101, 3533: 158, 3691: 192, 3883: 107, 3990: 151, 4141: 196, 4337: 240, 4577: 112, 4689: 127, 4816: 119, 4935: 171, 5106: 468, 5574: 682, 6256: 133, 6389: 114, 6503: 196, 6699: 226, 6925: 132, 7057: 205, 7262: 194, 7456: 179, 7635: 181, 7816: 152, 7968: 221, 8189: 186, 8375: 149, 8524: 195, 8719: 132, 8851: 305, 9156: 269, 9425: 171, 9596: 173, 9769: 153} c = 200
        #                               {1460: 234, 1694: 207, 1901: 204, 2105: 201, 2306: 166, 2472: 94, 2566: 455, 3021: 100, 3121: 206, 3327: 104, 3431: 262, 3693: 192, 3885: 101, 3986: 153, 4139: 198, 4337: 140, 4477: 339, 4816: 612, 5428: 147, 5575: 679, 6254: 249, 6503: 204, 6707: 558, 7265: 192, 7457: 178, 7635: 184, 7819: 151, 7970: 219, 8189: 188, 8377: 146, 8523: 197, 8720: 437, 9157: 98, 9255: 169, 9424: 175, 9599: 171, 9770: 152} 平滑后的数据
        # peaksList.append(peaks)
        # peaksList.append(peaksCWT)
        # peaksList.append(peaks2)
        # peaksList.append(rawPeak)
        # dataList.append(peaks[0])
        # dataList.append(peaks[1])
        # ndrDictSTL = findNDR(s, x, contig, peaks, residualDatam, rawDataList, label = 'STL处理数据', color = 'r', smoothMethod= 'STL', peakDisThreshold = 280)
        # ndrInformation_Smooth = findNDR(s, x, contig, peaks, peakObjectList, adjustWpsList, rawDataList,
        #                                 label='平滑算法数据', color='b',
        #                                 smoothMethod='平滑算法', peakDisThreshold=230)
        ndrObjectList = findNDR(x, start, contig, peakObjectList, adjustWpsList, rawDataList,
                                                       label='平滑算法数据', color='b',
                                                       smoothMethod='平滑算法', peakDisThreshold=230)
        # print(ndrInformation_Smooth['startPos'][0])
        # drawNDR(ndrInformation_Smooth, ndrObjectList, x, s, rawDataList, adjustWpsList)
        # writeToNDRFile(ndrInformation_Smooth)
        # peaks = scipy_signal_find_peaks_cwt(smoothData)
        # peaksList.append(peaks)

        # savgol_filter_func(wpsList)
        # diffList = diff(peaks[1], 1).tolist()
        # print(diffList)
        # peakDis.extend(diffList)

        # drawPeaks(peaksList, win, x, start)
        # drawSinglePeak(adjustWpsList, rawDataList, peakObjectList, s)
        # peaksList = []

        # drawPeaksBycwt(peaksList, win, x, s)

        # drawPeaksLength(peakDis)
        gc.collect()
        # writToExcel(dataList) #第一行为对原始数据进行标准化之后的新数据，第二行为FilterData，
        # 第三行为peaksX，第四行为peaks的属性
    # KernelDensityEstimate(kernel='linear', bandwidth=3, dataList=peakWdith, start=70, end=230, point=10000,
    #                       bin=80, maxYlim=0.025, minYlim=-0.005, title='波形宽度分布')
    # KernelDensityEstimate(kernel='linear', bandwidth=3, dataList=peakDis, start=50, end=400, point=10000,
    #                       bin=150, maxYlim=0.015, minYlim=-0.005, title='波形间距分布')
    t2 = time.clock()
    print("run time:%f s" % (t2 - t1))