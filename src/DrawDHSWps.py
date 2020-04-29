from WpsCal import *
import pysam
from scipy.signal import *
from scipy.interpolate import *
from DrawTSSWps import *

def getDHSData(pointFilePath, cnt):
    pointList = []
    file = open(pointFilePath, 'r')
    dataList = []
    while True:
        line = file.readline()
        if line == None or cnt < 0:
            break
        cnt -= 1
        data =line[:-1].split('\t')
        if len(data) == 1:
            break
        dataList.append(data)
    for data in dataList:
        pointList.append([str(data[0]), int(data[1]), int(data[2])])
    return pointList

def drawWPSWithDHS(dhs, dataList, y_label, x):
    figure, axes = plt.subplots(len(dataList), 1, figsize=(14, 14))
    # title_str = contig + '_' + str(start) + '_' + str(end) + '_' + point[3]
    lineList = []
    i = 0
    colorsList = ['b', 'r', 'k', 'g', 'c', 'm', 'y', 'w']
    for data in dataList:
        # axes[0].set_title(contig + '_' + str(start) + '_' + str(end) + '_' + point[3], fontsize=14)
        line = axes[i].plot(x[:min(len(x), len(data))], data[:min(len(x), len(data))], color=colorsList[i])
        lineList.append(line)
        axes[i].vlines([x[int(len(x) / 2)], int(dhs[1]), int(dhs[2])], ymin=np.min(data), ymax=np.max(data), color='r',
                       linestyles='dashed')
        i += 1

    # plt.legend([lineList[0][0], lineList[1][0], lineList[2][0], lineList[3][0], lineList[4][0], lineList[5][0], lineList[6][0]], y_label, loc='lower right')
    index = 0
    for axi in axes.ravel():
        axi.grid(axis="y", linestyle='--')
        axi.set_ylabel(y_label[index])
        axi.get_xaxis().get_major_formatter().set_useOffset(False)  # 去除科学计数法
    plt.show()





if __name__ == '__main__':
    OvarianPathList = ['/mnt/X500/farmers/chenhx/02.project/hexiaoti/01.sWGS_data/Ovarian.all.bam']
    NormalPathList = ['/mnt/X500/farmers/chenhx/02.project/hexiaoti/02.cellPaper_data/IH01.bam',
                      '/mnt/X500/farmers/chenhx/02.project/hexiaoti/02.cellPaper_data/BH01.bam',
                      '/mnt/X500/farmers/chenhx/02.project/hexiaoti/02.cellPaper_data/IH02.bam']
    # DHSPointList = getDHSData('/home/chenlb/WPSCalProject/filePathList/iDOCRaSE.chr_n.bed', 10000)
    wgsPointFilePath = '/mnt/X500/farmers/chenhx/02.project/hexiaoti/05.NDR_all/wholegenome.20k.filter.bin'
    wgsPointList = getDHSData(wgsPointFilePath, 1000000000)
    allPoint = len(wgsPointList)
    round = 0
    s = e = 0
    peaksList = []
    # kal_filter = Kalman_filter.kalman_filter(Q=0.1, R=10)
    # chr12 34484000 34561000
    t1 = time.clock()
    for wgsPoint in wgsPointList:
        print('*************************************  round : ', round, '  ->  ', allPoint,
              ' *************************************')
        round += 1
        contig = wgsPoint[0]
        start = int(wgsPoint[1]) - 1500
        end = int(wgsPoint[2]) + 1500
        step = end - start
        peaksList = []
        dataList = []
        x = np.arange(start, end)
        # wpsList_Nor = np.zeros(step + 600, dtype=np.int)
        # wpsList_Ovarian = np.zeros(step + 600, dtype=np.int)
        length = step + 1
        # wpsList_Nor, coverageArray_Nor = getWpsListAndCover(NormalPathList, wpsList_Nor, win, contig, start, end, s, e)
        # wpsList_Ovarian, coverageArray_Ovarian = getWpsListAndCover(OvarianPathList, wpsList_Ovarian, win, contig, start, end, s, e)
        wpsList_Nor, lFdepth_Nor, sFdepth_Nor = callOneBed(NormalPathList, contig, start, end, win=120)
        rawWPS = np.array(wpsList_Nor)
        adjustWpsList_Nor = AdjustWPS(wpsList_Nor)

        peaks = scipy_signal_find_peaks(adjustWpsList_Nor, height=0.01, distance=30, prominence=0.1, width=[25, 170])
        peakObjectList = getValley(adjustWpsList_Nor, rawWPS, np.array(peaks[1]), 4)
        peaks.append(peakObjectList)
        # fastFilter2(adjustWpsList)
        lowLength, varWidth, squareWave = fastFilter(adjustWpsList_Nor, peaks, False)
        if ((lowLength > 140 and varWidth > 280) or (lowLength > 150 or varWidth > 450)) and not (
                lowLength < 90 or varWidth < 180):
            # print('lowLength : ', lowLength, ' varWidth : ', varWidth)
            continue
        try:
            base = peakutils.baseline(adjustWpsList_Nor, 8)
        except ZeroDivisionError:  # 'ZeroDivisionError'除数等于0的报错方式
            continue
        adjustWpsList_Nor = np.subtract(adjustWpsList_Nor, base)
        # kalWpsList_Nor = kalman_filter(kal_filter, adjustWpsList_Nor)
        # smoothWpsList_Nor = smooth(adjustWpsList_Nor, 51)
        # smoothWpsList_Nor = medfilt(adjustWpsList_Nor, 25)
        # z = sm.nonparametric.lowess(adjustWpsList_Nor, np.array([i for i in range(len(wpsList_Nor))]), frac=0.003)
        # smoothWpsList_Nor = z[:, 1]
        smoothWpsList_Nor = savgol_filter_func(adjustWpsList_Nor, 35, 1)
        # waveWpsList_Nor = waveletSmooth(adjustWpsList_Nor, 0.1)
        norm_lFdepth_Nor = preprocessing.minmax_scale(lFdepth_Nor)
        peakHeight = []
        for data in [smoothWpsList_Nor]:
            peaks = scipy_signal_find_peaks(data, height=0.28, distance=25, prominence=0.25, width=[25, 170])
            peakObjectList = getValley(data, rawWPS, peaks[1], 5)
            # peakObjectListCWT = getValley(adjustWpsList, rawDataList, peaksCWT[1], 4)
            #
            mergeValley(peakObjectList)
            # mergeValley(peakObjectListCWT)
            peakObjectList = mergePeaks(peakObjectList)
            peaksList.append([data, peaks[1], peakObjectList])
            # for i in range(len(peakObjectList) - 1):
            #     width = peakObjectList[i].endPos - peakObjectList[i].startPos
            #     if width > 80 and width < 230:
            #         peakWdith.append(width)
            #     peakHeight.append(100 * min(smoothWpsList_Nor[peakObjectList[i].peakIndex] - smoothWpsList_Nor[peakObjectList[i].startPos] ,
            #                           smoothWpsList_Nor[peakObjectList[i].peakIndex] - smoothWpsList_Nor[peakObjectList[i].endPos]))
            #     peakDis.append(peakObjectList[i + 1].peakIndex - peakObjectList[i].peakIndex)
            #     peakDisSet.add(peakObjectList[i + 1].peakIndex - peakObjectList[i].peakIndex)
        ndrObjectList = findTssNDR(x, start, contig, peakObjectList, smoothWpsList_Nor, rawWPS, norm_lFdepth_Nor,squareWave,
                                   label='sg滤波数据', color='b',
                                   smoothMethod='sg滤波数据', peakDisThreshold=230)

        # dataList = [lFdepth_Nor, rawWPS, base, smoothWpsList_Nor]
        # drawWPSWithDHS(wgsPoint, dataList, ['普通样品覆盖度', '原始WPS', '中值预处理WPS', '卡尔曼滤波', '卷积平滑',
        #                    'Savitzky-Golay滤波', '小波滤波'], x)
