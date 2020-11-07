# from WpsCal import *
# import pysam
import sys
import getopt
import numpy as np
import pysam
import scipy
from scipy.signal import *
from scipy.signal import savgol_filter
from scipy import interpolate
from scipy import integrate
# from pykalman import *
from scipy.signal import medfilt
from sklearn import *
from sklearn.neighbors import KernelDensity
import peakutils
from Peak import Peak
from NDR import NDR
import matplotlib as mpl
import matplotlib.pyplot as plt
from pylab import *
# from scipy.signal import *
# from scipy.interpolate import *
# from scipy import *

class gene:
    def __init__(self, name, chr, startPos, endPos, flag, tssBinStart, tssBinEnd):
        '''
        :param name: 基因名称
        :param chr: 染色体编号
        :param startPos: 基因起始位点 startPos和endPos的大小和正副链相关
        :param endPos: 基因终止位点
        :param flag: 正副链标志
        :param tssBinStart: tss起始位点
        :param tssBinEnd: tss终止位点
        '''
        self.name = name
        self.chr = chr
        self.startPos = startPos
        self.endPos = endPos
        self.flag = flag
        self.endPos = endPos
        self.tssBinStart = tssBinStart
        self.tssBinEnd = tssBinEnd

    def __str__(self):
        return 'name : %s, chr : %s, startPos : %d, endPos : %d, flag : %s' % (
            self.name, self.chr, self.startPos, self.endPos, self.flag)


def getTSSPoint(filepath, cnt):
    geneDict = {}
    with open(filepath, mode='r') as f:
        while True:
            line = f.readline()
            if line == None or len(line) <= 1 or cnt < 0:
                break
            cnt -= 1
            temp = line[:-1].split('\t')
            if len(temp) < 4 or temp[0] == 'NULL':
                continue
            if temp[3] == '+':
                geneDict[temp[4]] = gene(name=temp[4], chr=temp[0], startPos=int(temp[1]), endPos=int(temp[2]),
                                         flag=temp[3], tssBinStart=int(temp[1]) - 2500, tssBinEnd=int(temp[1]) + 2500)
            else:
                geneDict[temp[4]] = gene(name=temp[4], chr=temp[0], startPos=int(temp[1]), endPos=int(temp[2]),
                                         flag=temp[3], tssBinStart=int(temp[2]) - 2500, tssBinEnd=int(temp[2]) + 2500)
            # pointList.append([temp[0], int(temp[1]) - 1000, int(temp[2]) + 1000, temp[3]])
    return geneDict


def getCover(coverageAllArray, bamfile, contig, start, end):
    for pileupcolumn in bamfile.pileup(contig, start, end):
        if pileupcolumn.pos - start >= len(coverageAllArray):
            break;
        # print("\ncoverage at base %s = %s" %
        #       (pileupcolumn.pos, pileupcolumn.n))
        for pileupread in pileupcolumn.pileups:
            if not pileupread.is_del and not pileupread.is_refskip and abs(pileupread.alignment.isize) > 120 and abs(
                    pileupread.alignment.isize) < 200:
                coverageAllArray[pileupcolumn.pos - start] += 1
    return coverageAllArray


def judgeLowDepth(depth, startPos, endPos):
    ndrWin = 300
    smallNdrWin = 100
    depSum = np.zeros(len(depth) + 1)
    depSum[1:] = np.cumsum(depth)  # depth[i - > j] = depSum[j + 1] - depSum[i]
    # depthList = []
    ndrAreaDepth = 10000
    smallNdrAreaDepth = 10000

    startPos = max(startPos, ndrWin)
    endPos = min(endPos, len(depth) - ndrWin)
    minIndex = startPos
    for i in range(startPos, endPos):
        if depSum[i + ndrWin] - depSum[i - ndrWin] < ndrAreaDepth:
            ndrAreaDepth = depSum[i + ndrWin] - depSum[i - ndrWin]
            minIndex = i
        smallNdrAreaDepth = min(depSum[i + smallNdrWin] - depSum[i - smallNdrWin], smallNdrAreaDepth)
    # for i in range(ndrWin, len(depth) - ndrWin, 10):
    #     allDepth = depSum[i + ndrWin] - depSum[i - ndrWin]
    #     depthList.append(allDepth)
    # if ndrAreaDepth <= 200:
    #     KernelDensityEstimate(kernel='gaussian', bandwidth=10, dataList=depthList, start=50, end=520, point=10000,
    #                           bin=80, maxYlim=0.012, minYlim=-0.003,
    #                           title='600bp区间Depth总和分布 -- ndr区间depth' + str(ndrAreaDepth))
    return ndrAreaDepth, smallNdrAreaDepth, minIndex


def judgeNDRWithDepth(smoothData, rawDataList, depth, squareWave, ndr, peakObjectList, contig, start, slidWinSize,
                      flag):
    peakLeftIndex = trackLeftPeak(smoothData, rawDataList, ndr, slidWinSize=slidWinSize)
    peakRightIndex = trackRightPeak(smoothData, rawDataList, ndr, slidWinSize=slidWinSize)
    ndrMax = 0
    if peakRightIndex != peakLeftIndex:
        ndrMax = np.max(smoothData[min(peakRightIndex, peakLeftIndex): max(peakLeftIndex, peakRightIndex)])
        # print('ndr sum : ', (ndrMax - min(smoothData[peakRightIndex], smoothData[peakLeftIndex])))
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
    ndrArea = getTriangleArea([float(xPeak[0]), smoothData[xPeak[0]]], [float(topPeak), smoothData[topPeak]],
                              [float(xPeak[3]), smoothData[xPeak[3]]])
    if flag:
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(14, 8))
        xStrat = max(ndr.startPos - 2500, 0)
        xEnd = min(ndr.endPos + 2500, min(len(smoothData), len(rawDataList)))
        x = np.array([i + start for i in range(xStrat, xEnd)])
        axes[0].plot(x, depth[xStrat: xEnd], 'k')
        line1 = axes[1].plot(x, smoothData[xStrat: xEnd], 'r')
        axes[1].plot(xPeak + start, np.array(smoothData)[xPeak], 'k')
        line2 = axes[2].plot(x, rawDataList[xStrat: xEnd], 'b')

        # ax1.vlines(x=peaksCorrectionX[i] + 30, ymin=yminPos - 0.2, ymax=ymaxPos + 0.2, colors='k', linestyles='dashed')

        axes[2].plot(xPeak + start, rawDataList[xPeak], 'k')
        axes[1].plot(xPeak + start, smoothData[xPeak], 'xk')
        axes[2].plot(xPeak + start, np.array(rawDataList)[xPeak], 'xk')
        axes[2].plot(np.array([xPeak[0], xPeak[3]]) + start, rawDataList[np.array([xPeak[0], xPeak[3]])], 'r')
        axes[1].plot(np.array([xPeak[0], xPeak[3]]) + start, smoothData[np.array([xPeak[0], xPeak[3]])], 'r')
        axes[1].set_title(
            'ndr.aveHeight : ' + str(get_two_float(ndr.aveHeight, 2)) + '  kLeft : ' + str(
                get_two_float(kLeft, 2)) + '  kRight : ' + str(get_two_float(kRight, 2))
            + ' kNDR : ' + str(get_two_float(kNDR, 2)) + ' ndrArea : ' + str(get_two_float(ndrArea, 2)))

        list = np.array([ndr.startPos, ndr.endPos])
        axes[1].vlines(x=list + start, ymin=smoothData[list] - 0.15,
                       ymax=smoothData[list] + 0.15, colors='k', linestyles='dashed')

        axes[2].vlines(x=list + start, ymin=rawDataList[list] - abs(rawDataList[list] * 1.4),
                       ymax=rawDataList[list] + abs(rawDataList[list] * 1.4), colors='k', linestyles='dashed')
        #
        plt.legend((line1[0], line2[0]), ('平滑后的WPS曲线', '原始WPS曲线'), loc='best')
        axes[2].set_title('疑似NDR区域')
        plt.xlabel(str(contig) + '号染色体基因组位点')
        axes[1].set_ylabel('标准化wps值')
        plt.ylabel('wps值')
        plt.show()
    # print('judgeNDR done')
    return kLeft, kRight, kNDR, ndrArea, ndrMax


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
        if 1 + peak.leftK * peak.rightK != 0:
            peakAngel[i] = math.atan(abs((peak.leftK - peak.rightK) / (1 + peak.leftK * peak.rightK))) * 180 / 3.1415
        else:
            peakAngel[i] = 90
        peakArea[i] = getTriangleArea([float(peak.startPos), rawDataList[peak.startPos]],
                                      [float(peak.peakIndex), rawDataList[peak.peakIndex]],
                                      [float(peak.endPos), rawDataList[peak.endPos]]) / 100
        if i < peakListSize - 1 and nearPeakList[i + 1].peakIndex - peak.peakIndex < 300 and maxDisPeakCount < 5:
            peakDis.append(nearPeakList[i + 1].peakIndex - peak.peakIndex)
        elif i < peakListSize - 1 and nearPeakList[i + 1].peakIndex - peak.peakIndex > 360:
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
            'width var : ' + get_two_float(str(varWidth), 2) + '   height var : ' + get_two_float(str(varHeight), 2)
            + '   angel var : ' + get_two_float(str(varAngel), 2) +
            '   varDis : ' + get_two_float(str(varDis), 2) + '  varArea : ' + get_two_float(str(varArea), 2))
        axes[0].plot(nearPeakX, smoothData[nearPeakX], 'xr')
        axes[1].plot(nearPeakX, rawDataList[nearPeakX], 'xb')
        axes[0].vlines(x=peakVell, ymin=smoothData[peakVell] - 0.2, ymax=smoothData[peakVell] + 0.2)
        axes[1].vlines(x=peakVell, ymin=rawDataList[peakVell] - 30, ymax=rawDataList[peakVell] + 30)
        plt.show()
        print('width var : ', varWidth, ' width : ', peakWidth)
        print('height var : ', varHeight, ' height : ', peakHeight)
        print('angel var : ', varAngel, ' angel : ', peakAngel)
        print('Dis var : ', varDis, ' Dis : ', peakDis)
    # print('haveNearContinuouslyPeak done')
    return varWidth, varHeight, varAngel, varDis, varArea, len(peakRightList) + len(peakLeftList) - maxDisPeakCount

def get_two_float(f_str, n):
    f_str = str(f_str)      # f_str = '{}'.format(f_str) 也可以转换为字符串
    a, b, c = f_str.partition('.')
    c = (c+"0"*n)[:n]       # 如论传入的函数有几位小数，在字符串后面都添加n为小数0
    return ".".join([a, c])

def findTssNDR(x, start, contig, peakObjectList, smoothData, rawDataList, depth, squareWave, label, color, smoothMethod,
               peakDisThreshold):
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
    ndrObjectList = []
    for i in range(len(peakObjectList) - 1):
        peakHeight1 = smoothData[peakObjectList[i].peakIndex] - max(smoothData[peakObjectList[i].startPos],
                                                                    smoothData[peakObjectList[i].endPos])
        peakHeight2 = smoothData[peakObjectList[i + 1].peakIndex] - max(smoothData[peakObjectList[i + 1].startPos],
                                                                        smoothData[peakObjectList[i + 1].endPos])
        if (peakObjectList[i + 1].startPos - peakObjectList[i].endPos > 135 or peakObjectList[i + 1].peakIndex -
            peakObjectList[i].peakIndex > 280 and peakObjectList[i + 1].peakIndex - peakObjectList[
                i].peakIndex < 1500) or (peakHeight1 < 0.3 and peakHeight2 < 0.3 and peakObjectList[i + 1].peakIndex -
                                        peakObjectList[i].peakIndex > 240):
            if peakObjectList[i + 1].startPos != peakObjectList[i].endPos:
                aveHeight = getPeakAveHeight(smoothData, peakObjectList[i].endPos,
                                             peakObjectList[i + 1].startPos) / (
                                    peakObjectList[i + 1].startPos - peakObjectList[i].endPos)
                if aveHeight != 0:
                    ndr = NDR(peakObjectList[i].endPos, peakObjectList[i + 1].startPos,
                              peakObjectList[i + 1].startPos - peakObjectList[i].endPos)
                    ndr.aveHeight = aveHeight
                    ndrAreaDepth, smallNdrAreaDepth, minIndex = judgeLowDepth(depth, ndr.startPos, ndr.endPos)
                    # print('ndrAreaDepth -- smallNdrAreaDepth', ndrAreaDepth, ' -- ', smallNdrAreaDepth)
                    if ndrAreaDepth > 250 or smallNdrAreaDepth > 35:
                        continue
                    kRight, kLeft, kNDR, ndrArea, ndrMax = judgeNDRWithDepth(smoothData, rawDataList, depth, squareWave,
                                                                             ndr,
                                                                             peakObjectList, contig, start,
                                                                             slidWinSize=6,
                                                                             flag=False)
                    coef, mse = linearJudgeNDR(smoothData, rawDataList, ndr, False)
                    # print([coef, mse])
                    varWidth, varHeight, varAngel, varDis, varArea, peakCount = haveNearContinuouslyPeak(smoothData,
                                                                                                         rawDataList,
                                                                                                         peakObjectList,
                                                                                                         5, ndr,
                                                                                                         False)
                    condition = [varWidth < 450, varHeight < 1200, varAngel < 220, varDis < 350, varArea < 300,
                                 peakCount >= 7]
                    cnt = condition.count(True)
                    if (ndrAreaDepth > 250 or (cnt < 4 and varDis >= 250 and varAngel >= 100 and varWidth >= 200 and varArea >= 100) or varDis >= 700 or varAngel >= 450 or varWidth >= 800) and not (
                          kLeft < 0.22 and kRight < 0.22 and kNDR < 0.25 and ndrArea < 10 and coef[0][
                        0] * 1000 < 3 and mse < 4 and smallNdrAreaDepth < 45):
                        continue
                    ndr.startPos = minIndex - 300
                    ndr.endPos = minIndex + 300
                    ndrObjectList.append(ndr)
                    print('find a NDR')
    writeOCRsToFile(contig, start, ndrObjectList)
    return ndrObjectList



def writeOCRsToFile(contig, start, ndrObjectList):
    # with open('ndrInfo.txt', mode='a+') as f:
    #     for ndr in ndrObjectList:
    #         list = [contig, start + ndr.startPos, start + ndr.endPos, start]
    #         list.extend(rawDataList[max(ndr.startPos - 1000, 0) : min(ndr.endPos + 1000, len(rawDataList))])
    #         f.write(str(list)+ '\n')
    with open('ndr_DHSAndTSS.chr10_e.txt', mode='a+') as f:
        for ndr in ndrObjectList:
            list = str(contig) + "\t" + str(start + ndr.startPos) + "\t" + str(start + ndr.endPos)
            f.write(str(list)+ '\n')

def callOneBed(pathList, contig, bed1, bed2, win):
    # tmpInfor = tmpBed.split("\t")

    bed1 = bed1 - win
    bed2 = bed2 + win
    length = bed2 - bed1 + 1
    array = np.zeros(length, dtype=np.int)
    depth = np.zeros(length, dtype=np.int)
    depth2 = np.zeros(length, dtype=np.int)
    #################
    for filePath in pathList:
        bamfile = readFileData(filePath)
        # print(filePath)
        for r in bamfile.fetch(contig, bed1, bed2):
            # if (not r.is_reverse) and (not r.is_unmapped) and (not r.mate_is_unmapped) and r.mate_is_reverse :
            if (not r.is_reverse) and (not r.is_unmapped) and (not r.mate_is_unmapped) and r.mate_is_reverse:
                # print(r.reference_name,r.isize,r.reference_start,r.reference_start+r.isize)
                if r.isize >= 35 and r.isize <= 80:
                    start = r.reference_start - bed1
                    end = r.reference_start + r.isize - bed1
                    # depth + 1
                    dstart = start
                    dend = end
                    if dstart < 0:
                        dstart = 0
                    if dend > length:
                        dend = length
                    d = dstart
                    while d < dend:
                        depth2[d] += 1
                        d += 1

                if r.isize < win or r.isize > 180:
                    continue
                start = r.reference_start - bed1
                end = r.reference_start + r.isize - bed1
                # depth + 1
                dstart = start
                dend = end
                if dstart < 0:
                    dstart = 0
                if dend >= length:
                    dend = length
                d = dstart
                while d < dend:
                    depth[d] += 1
                    d += 1

                # [$start+W/2,$end-W/2] WPS+1
                region1 = start + int(win / 2)
                region2 = end - int(win / 2)
                if region1 < 0:
                    region1 = 0
                if region2 > length:
                    region2 = length
                i = region1
                while i < region2:
                    array[i] += 1
                    i += 1
                # [$start-w/2,$start-1+w/2] WPS-1
                region1 = start - int(win / 2)
                region2 = start + int(win / 2) + 1
                if region1 < 0:
                    region1 = 0
                if region2 > length:
                    region2 = length
                i = region1
                while i < region2:
                    array[i] -= 1
                    i += 1
                # [end-w/2+1,$end+w/2] WPS-1
                region1 = end - int(win / 2) + 1
                region2 = end + int(win / 2)
                if region1 < 0:
                    region1 = 0
                if region2 > length:
                    region2 = length
                i = region1
                while i < region2:
                    array[i] -= 1
                    i += 1
        # adjustWPS = AdjustWPS(array)
    lenth1 = len(array) - win - 1
    bed1 += win
    bed2 -= win
    array = np.array(array[win: lenth1], dtype=np.float)
    depth = depth[win: lenth1]
    depth2 = depth2[win: lenth1]
    return array, depth, depth2


def getMinMax(array):
    minNum = 100000
    maxNum = -100000
    for num in array:
        min = min(minNum, num)
        maxNum = min(maxNum, num)
    return minNum, maxNum


def drawWPS(dataList, y_label, x):
    figure, axes = plt.subplots(len(dataList), 1, figsize=(14, 14))
    # title_str = contig + '_' + str(start) + '_' + str(end) + '_' + point[3]
    lineList = []
    i = 0
    colorsList = ['b', 'r', 'k', 'g', 'c', 'm', 'y', 'w']
    for data in dataList:
        # axes[0].set_title(contig + '_' + str(start) + '_' + str(end) + '_' + point[3], fontsize=14)
        line = axes[i].plot(x[:min(len(x), len(data))], data[:min(len(x), len(data))], color=colorsList[i])
        lineList.append(line)
        axes[i].vlines(x[int(len(x) / 2)], ymin=np.min(data), ymax=np.max(data), color='r',
                       linestyles='dashed')
        i += 1

    # plt.legend([lineList[0][0], lineList[1][0], lineList[2][0], lineList[3][0], lineList[4][0], lineList[5][0], lineList[6][0]], y_label, loc='lower right')
    index = 0
    for axi in axes.ravel():
        axi.grid(axis="y", linestyle='--')
        axi.set_ylabel(y_label[index])
        axi.get_xaxis().get_major_formatter().set_useOffset(False)  # 去除科学计数法
    plt.show()


def drawPeaksWithDepth(lFDepth, base, dataList, win, x, start, y_label):
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
    fig, axes = plt.subplots(len(dataList) + 0, 1, figsize=(8, int((len(dataList) + 0) * 2.5)), dpi=200)
    # axes[0].plot(x[0: min(len(x), len(lFDepth))], lFDepth[0: min(len(x), len(lFDepth))], colorsList[0])
    # axes[0].grid(axis="y", linestyle='--')
    # axes[1].plot(x[0: min(len(x), len(base))], base[0: min(len(x), len(base))], '#12aa9c')
    # axes[1].grid(axis="y", linestyle='--')
    for data in dataList:
        wpsFilterData = np.array(data[0])
        peaksX = data[1]
        peakObjectList = data[2]
        minLen = min(len(x), len(wpsFilterData))
        print('minLen = ', minLen)
        line = axes[cIndex].plot(x[0: minLen], wpsFilterData[0: minLen], colorsList[cIndex])
        lineList.append(line)
        axes[cIndex].plot(start + peaksX, wpsFilterData[peaksX], 'x' + colorsList[(cIndex + 2) % len(colorsList)])
        axes[cIndex].grid(axis="y", linestyle='--')
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
            cIndex = (cIndex + 1) % len(colorsList)
            continue
        ymaxWps = np.max(wpsFilterData[vX]) * 0.8
        yminWps = np.min(wpsFilterData[vX]) * 0.8
        axes[cIndex].vlines(x=vX + start,
                            ymin = yminWps,
                            ymax= ymaxWps, color='k',
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
        #
    # plt.legend([lineList[0][0], lineList[1][0], lineList[2][0], lineLis     t[3][0], lineList[4][0]], y_label,
    #            loc='lower right')
    plt.show()

def getPathList(bamListPath):
    LungPathList = []
    for lungPath in open(bamListPath):
        if lungPath == None or len(lungPath) == 0:
            break
        LungPathList.append(lungPath[:-1])
    return LungPathList

def baseline_als(y, lam, p, niter=10):
    s  = len(y)
    # assemble difference matrix
    D0 = scipy.sparse.eye( s )
    d1 = [np.ones( s-1 ) * -2]
    D1 = scipy.sparse.diags( d1, [-1] )
    d2 = [np.ones( s-2 ) * 1]
    D2 = scipy.sparse.diags( d2, [-2] )
    D  = D0 + D2 + D1
    w  = np.ones(s)
    for i in range( niter ):
        W = scipy.sparse.diags([w], [0])
        Z =  W + lam*D.dot( D.transpose())
        z = scipy.sparse.linalg.spsolve( Z, w*y )
        w = p * (y > z) + (1-p) * (y < z)
    # fig, axes = plt.subplots(3, 1)
    # axes[0].plot(y)
    # axes[1].plot(np.subtract(y, z))
    # axes[2].plot(z)
    # plt.show()
    return z

def baseline_als2(y, lam, p, niter=10):
  L = len(y)
  D = scipy.sparse.csc_matrix(np.diff(np.eye(L), 2))
  w = np.ones(L)
  for i in range(niter):
    W = scipy.sparse.spdiags(w, 0, L, L)
    Z = W + lam * D.dot(D.transpose())
    z = scipy.sparse.linalg.spsolve(Z, w*y)
    w = p * (y > z) + (1-p) * (y < z)
  return z

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

def readBamFileList(filePath):
    bamfileList = []
    with open(filePath, 'r') as bamListReader:
        while True:
            path = bamListReader.readline().replace('\n', '')
            if not path:
                break
            bamfileList.append(path)
    return bamfileList

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

def getMMM(tmpArray):
    tmpArray = np.array(tmpArray)
    return np.min(tmpArray), np.median(tmpArray), np.max(tmpArray)

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
    return peaks

def savgol_filter_func(wpsList, filterWin, poly):
    '''
    SG Filter
    SG滤波算法平滑波形,多项式平滑算法 Savitzky-Golay平滑算法
    :param wpsList: wpsList
    :return: wpsListFiler SG滤波之后的WPS数据 (WPS data after SG filtering)
    '''

    x = [i for i in range(len(wpsList))]
    x = np.array(x)  # list to ndarray
    wpsList = np.array(wpsList)
    wpsFilter = savgol_filter(wpsList, filterWin, poly)  # windows length(int)： (must be a positive odd integer);
    # polyorder(int)；The order of the polynomial used to fit the samples. polyorder must be less than window_length.
    print('savgol filter done')
    return wpsFilter

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

def getPointDis(point0, point1):
    return ((point0[0] - point1[0]) ** 2 + (point0[1] - point1[1]) ** 2) ** 0.5

def getTriangleArea(point0, point1, point2):
    line0 = getPointDis(point0, point1)
    line1 = getPointDis(point0, point2)
    line2 = getPointDis(point1, point2)
    p = (line0 + line1 + line2) / 2
    return (p*(p - line0)*(p - line1)*(p - line2))** 0.5

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

def getValley(wpsList, rawDataList, peaks, slidWinSize):
    leftValleyList = []
    rightValleyList = []
    for peakX in peaks:
        left = track2leftValley(wpsList, peakX, slidWinSize)
        right = track2rightValley(wpsList, peakX, slidWinSize)
        leftValleyList.append(left)
        rightValleyList.append(right)
    peakObjectList = []

    for i in range(len(peaks)):
        leftPeakK = 0
        rightPeakK = 0
        if peaks[i] - leftValleyList[i] != 0 and rawDataList[peaks[i]] != rawDataList[leftValleyList[i]]:
            leftPeakK = (rawDataList[peaks[i]] - rawDataList[leftValleyList[i]]) / (peaks[i] - leftValleyList[i])
        if peaks[i] - rightValleyList[i] != 0 and rawDataList[peaks[i]] != rawDataList[rightValleyList[i]]:
            rightPeakK = (rawDataList[peaks[i]] - rawDataList[rightValleyList[i]]) / (peaks[i] - rightValleyList[i])
        peakObjectList.append(
            Peak(peaks[i], leftValleyList[i], rightValleyList[i], rightValleyList[i] - leftValleyList[i], leftPeakK, rightPeakK))
    return peakObjectList


def mergeValley(peakObjectList):
    for i in range(len(peakObjectList)):
        if i == len(peakObjectList) - 1:
            break;
        if peakObjectList[i].endPos > peakObjectList[i + 1].peakIndex:
            peakObjectList[i].endPos = int((peakObjectList[i].peakIndex + peakObjectList[i + 1].peakIndex) / 2)
        elif peakObjectList[i + 1].startPos < peakObjectList[i].endPos:
            peakObjectList[i + 1].startPos = peakObjectList[i].endPos
    return peakObjectList

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


inputFilePath = ''
outputFilePath = ''

if __name__ == '__main__':
    '''
        ndr:nucleosome-depleted region
        ocr:Open chromatin region
        In this program, the meaning of ndr is the same as ocr
    '''
    opts, args = getopt.getopt(sys.argv[1:], "hi:o:", ["help", "input=", "output="])
    for o, a in opts:
        if o in ("-h", "--help"):
            # usage
            usage = '''Usage:   python OCRDetectBycfDNA.py [-h usage] [-i input file] [-o OCRs output bed]
    Example: python OCRDetectBycfDNA.py -i bamFileList.txt -o OCRs.bed
    Options:
        -h: usage
        -i: bam file list of cfDNA
        -o: output file of detected OCRs'''
            print(usage)
            sys.exit()
        if o in ("-i", "--input"):
            inputFilePath = a
        if o in ("-o", "--output"):
            outputFilePath = a
    bamfileList = readBamFileList(inputFilePath)

    pointFilePath = './wholegenome.20k.filter.bin'
    pointList = getPointData(pointFilePath, 1000000000)
    allPoint = len(pointList)
    round = 0
    s = e = 0
    peaksList = []
    peakWdith = []
    peakDis = []
    for point in pointList:
        print('*************************************  round : ', round, '  ->  ', allPoint,
              ' *************************************')
        round += 1
        contig = point[0] #if the format of contig is chr*,  contig = 'chr' + point[0]
        start = int(point[1])
        end = int(point[2])
        step = end - start
        peaksList = []
        dataList = []
        x = np.arange(start, end)
        length = step + 1
        wpsList_Nor, lFdepth_Nor, sFdepth_Nor = callOneBed(bamfileList, contig, start, end, win=120)
        rawWPS = np.array(wpsList_Nor)
        adjustWpsList_Nor = AdjustWPS(wpsList_Nor)
        squareWave = []
        try:
            base = peakutils.baseline(adjustWpsList_Nor, 8)
        except ZeroDivisionError:  # 'ZeroDivisionError'除数等于0的报错方式^M
            base = np.zeros(len(adjustWpsList_Nor))
        adjustWpsList_Nor = np.subtract(adjustWpsList_Nor, base)
        smoothWpsList_Nor = savgol_filter_func(adjustWpsList_Nor, 35, 1) #SG Filter
        norm_lFdepth_Nor = preprocessing.minmax_scale(lFdepth_Nor)
        peakHeight = []
        for data in [smoothWpsList_Nor]:
            peaks = scipy_signal_find_peaks(data, height=0.28, distance=25, prominence=0.25, width=[25, 170])
            peakObjectList = getValley(data, rawWPS, peaks[1], 5)
            mergeValley(peakObjectList)
            peakObjectList = mergePeaks(peakObjectList)
            peaksList.append([data, peaks[1], peakObjectList])
        ndrObjectList = findTssNDR(x, start, contig, peakObjectList, smoothWpsList_Nor, rawWPS, norm_lFdepth_Nor,
                                   squareWave,
                                   label='sg滤波数据', color='b',
                                   smoothMethod='sg滤波数据', peakDisThreshold=230)
