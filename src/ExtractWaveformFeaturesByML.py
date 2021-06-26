import WpsCal as wpc
from WpsCal import *
from scipy import signal
import pandas as pd
from DrawTSSWps import *
from sklearn.preprocessing import  StandardScaler, MinMaxScaler

from scipy import fftpack
import seaborn as sns
import warnings
import pickle
warnings.filterwarnings('ignore')

def average_fft(x, fft_size):
    n = len(x) // fft_size * fft_size
    tmp = x[:n].reshape(-1, fft_size)
    tmp *= signal.hann(fft_size, sym=0)
    xf = np.abs(np.fft.rfft(tmp) / fft_size)
    avgf = np.average(xf, axis=0)
    return 20 * np.log10(avgf)


def fft_combine(freqs, n, loops=1):
    length = len(freqs) * loops
    data = np.zeros(length)
    index = loops * np.arange(0, length, 1.0) / length * (2 * np.pi)
    for k, p in enumerate(freqs[:n]):
        if k != 0: p *= 2  # 除去直流成分之外，其余的系数都*2
        data += np.real(p) * np.cos(k * index)  # 余弦成分的系数为实数部
        data -= np.imag(p) * np.sin(k * index)  # 正弦成分的系数为负的虚数部
    return index, data


def fft(yt, sampling_rate, fft_size=None):
    '''
    :param yt: 输入数据
    :param sampling_rate: 采样率
    :param fft_size: fiter_size
    :return: freqs:
    '''
    if fft_size is None:
        fft_size = len(yt)
    yt = yt[:fft_size]
    yf = abs(np.fft.rfft(yt) / fft_size)
    freqs = np.linspace(0, 1.0 * sampling_rate / 2, 1.0 * fft_size / 2 + 1)
    return freqs, yf


def fftFilter(data, lowFreq, highFreq):
    B = 10.0
    Fs = 2 * B
    delta_f = 1
    N = len(data)
    T = N / Fs
    t = np.array([i for i in range(0, len(data))])
    dataFft = fftpack.fft(data)
    dataFreq = fftpack.fftfreq(n=len(data), d=1 / Fs)
    mask = np.where(dataFreq >= 0)
    if lowFreq != None:
        dataFft = dataFft * (abs(dataFreq) > lowFreq)
    if highFreq != None:
        dataFft = dataFft * (abs(dataFreq) < highFreq)
    dataFiltered = dataFft
    dataFftTFiltered = fftpack.ifft(dataFiltered)
    return dataFftTFiltered, mask


def drawPeaksWithDepth2(lFDepth, base, dataList, win, x, start, y_label):
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
    colorsList = ['gray', 'r', 'k', 'r', 'c', 'm', 'y', 'w']
    lineList = []
    cIndex = 0
    minLen = min(len(x), len(np.array(dataList[1][0])))
    plt.subplot(211)
    plt.plot(figsize=(12, 5))
    line1 = plt.plot(x[0: minLen], np.array(dataList[0][0])[0: minLen], color=colorsList[cIndex])
    plt.subplot(212)
    plt.plot(figsize=(12, 5))
    lineStyle = ['-', '-']
    add = [0.3, 0]
    for data in dataList:
        wpsFilterData = np.array(data[0])
        peaksX = data[1]
        peakObjectList = data[2]
        minLen = min(len(x), len(wpsFilterData))
        line2 = plt.plot(x[0: minLen], wpsFilterData[0: minLen] + add[cIndex], color=colorsList[cIndex],
                         linestyle=lineStyle[cIndex])
        # plt.plot(start + peaksX, wpsFilterData[peaksX], 'x' + colorsList[(cIndex + 2) % len(colorsList)])
        plt.grid(axis="y", linestyle='--')
        cIndex = (cIndex + 1) % len(colorsList)
    plt.xlabel('Genomic Coordinates')
    plt.ylabel('WPS(depth:5×)')
    plt.legend([line1[0], line2[0]], ['Original WPS Waveform', 'WPS Waveform after Savitzky Golay filtering'],
               loc='best')
    # plt.show()
    plt.savefig("/mnt/X500/farmers/chenlb/WpsImage/panel/hk_images/" + str(picIndex) + "_" + str(start) + "_" + str(
        end) + "_Chr" + contig + "_Ocr.jpg")
    plt.close()


def judgeLowDepth(depth, startPos, endPos):
    ndrWin = 300
    smallNdrWin = 100
    depSum = np.zeros(len(depth) + 1)
    depSum[1:] = np.cumsum(depth)  # depth[i - > j] = depSum[j + 1] - depSum[i]
    # depthList = []
    ndrAreaDepth = 1000000
    smallNdrAreaDepth = 1000000

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
    return ndrAreaDepth / (2 * ndrWin), smallNdrAreaDepth / (2 * smallNdrWin), minIndex

def haveNearContinuouslyPeak(smoothData, rawDataList, peakObjectList):
    '''
    :param smoothData:
    :param rawDataList:
    :param peakObjectList:
    :return: meanWidth, meanHeight, meanAngel, meanArea, varWidth, varHeight, varAngel, varArea
    '''
    peakListSize = len(peakObjectList)
    peakWidth = []
    peakHeight = []
    peakAngel = []
    peakArea = []
    nearPeakX = []
    peakVell = []
    maxDisPeakCount = 0
    for peak in peakObjectList:
        nearPeakX.append(peak.peakIndex)
        peakVell.append(peak.startPos)
        peakVell.append(peak.endPos)
        peakWidth.append(peak.width)
        peakHeight.append(rawDataList[peak.peakIndex] - min(rawDataList[peak.startPos], rawDataList[peak.endPos]))
        if 1 + peak.leftK * peak.rightK != 0:
            peakAngel.append(math.atan(abs((peak.leftK - peak.rightK) / (1 + peak.leftK * peak.rightK))) * 180 / 3.1415)
        else:
            peakAngel.append(90)
        peakArea.append(getTriangleArea([float(peak.startPos), rawDataList[peak.startPos]],
                                      [float(peak.peakIndex), rawDataList[peak.peakIndex]],
                                      [float(peak.endPos), rawDataList[peak.endPos]]) / 100)
    peakWidth = np.array(peakWidth)
    peakHeight = np.array(peakHeight)
    peakAngel = np.array(peakAngel)
    peakArea = np.array(peakArea)

    meanWidth = np.mean(peakWidth)
    meanHeight = np.mean(peakHeight)
    meanAngel = np.mean(peakAngel)
    meanArea = np.mean(peakArea)

    varWidth = np.var(peakWidth)
    varHeight = np.var(peakHeight)
    varAngel = np.var(peakAngel)
    varArea = np.var(peakArea)
    # print('haveNearContinuouslyPeak done')
    return meanWidth, meanHeight, meanAngel, meanArea, varWidth, varHeight, varAngel, varArea




def drawKde(dataList, labelList, savepath):
    # data1 = preprocessing.minmax_scale(data1)
    # data2 = preprocessing.minmax_scale(data2)
    # data3 = preprocessing.minmax_scale(data3)
    # data4 = preprocessing.minmax_scale(data4)
    for i in range(len(dataList)):
        ax = sns.distplot(dataList[i], kde_kws={"label": labelList[i]})
    # plt.ylim([0,0.1])
    fig = ax.get_figure()
    fig.savefig(savepath, bbox_inches='tight')
    plt.close()

def drawKde3(dataList, labelList, savepath, picIndex):
    # data1 = preprocessing.minmax_scale(data1)
    # data2 = preprocessing.minmax_scale(data2)
    # data3 = preprocessing.minmax_scale(data3)
    # data4 = preprocessing.minmax_scale(data4)
    for k in range(len(dataList[0][0])):
        for i in range(len(dataList)):
            c = []
            for j in range(len(dataList[i])):
                c.append(dataList[i][j][k])
            ax = sns.distplot(np.array(c), kde_kws={"label": labelList[i]})
        fig = ax.get_figure()
        fig.savefig(savepath + str(picIndex) + '_vector.jpg', bbox_inches='tight')
        picIndex = picIndex + 1
        plt.close()


def drawKde2(data1, data2, data3, data4, label1, label2, label3, label4, savepath):
    allData = []
    allData.extend(data1)
    allData.extend(data2)
    allData.extend(data3)
    allData.extend(data4)
    allData = np.array(allData)
    std = StandardScaler()
    allData_std = std.fit_transform(allData)
    data1 = allData_std[0:len(data1) - 1]
    data2 = allData_std[len(data1):len(data1) + len(data2) - 1]
    data3 = allData_std[len(data1) + len(data2):len(data1) + len(data2) + len(data3) - 1]
    data4 = allData_std[len(data1) + len(data2) + len(data3) :len(data1) + len(data2) + len(data3) + len(data4) - 1]
    # data3 = preprocessing.minmax_scale(data3)
    # data4 = preprocessing.minmax_scale(data4)
    ax = sns.distplot(data1, kde_kws={"label": label1})
    ax = sns.distplot(data2, kde_kws={"label": label2})
    ax = sns.distplot(data3, kde_kws={"label": label3})
    ax = sns.distplot(data4, kde_kws={"label": label4})
    plt.ylim(0.1)
    fig = ax.get_figure()
    fig.savefig(savepath, bbox_inches='tight')


def linearJudgeNDR(smoothData, rawDataList, startPos, endPos, picIndex, flag):
    model = linear_model.LinearRegression()
    x = np.array([i for i in range(startPos, endPos)])[:, np.newaxis]
    y = smoothData[startPos:endPos][:, np.newaxis]
    model.fit(x, y)
    if flag:
        # print(model.intercept_)  # 截距
        # print(model.coef_)
        plt.plot(x, y, color = 'k')
        plt.plot(x, model.coef_*x + model.intercept_, color = 'gray')
        title('直线为 ： ' + str(model.coef_) + 'x + ' + str(model.intercept_) + '    均方误差 : ' + str(np.mean((model.predict(x) - y) ** 2)))
        # plt.show()
        plt.savefig('/mnt/X500/farmers/chenlb/WpsImage/panel/linear'+str(picIndex + 1000)+'.jpg')
        picIndex += 1
        plt.close()
    return model, x, y

# def getAngleBisector(model1, model2):


def drawLinearLine(modelList, xList, yList):
    figure, axes = plt.subplots(3, 4, figsize=(3 * 4, 3 * 3))
    index = 0
    for j in range(4):
        for i in range(3):
            axes[i, j].plot(xList[index], yList[index], color='k')
            axes[i, j].plot(xList[index], modelList[index].coef_ * xList[index] + modelList[index].intercept_, color='gray')
            index += 1
    plt.savefig('/mnt/X500/farmers/chenlb/WpsImage/panel/linear' + str(picIndex + 1000) + '.jpg')
    plt.close()

def drawPeaksWithDepthForClassify(lFDepth, wpsFilterData, picIndex, win, x, start):
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
    cIndex = 2
    fig, axes = plt.subplots(2, 1, figsize=(4, 4))
    axes[0].axis('off')  # 去掉坐标轴
    axes[0].plot(x[0: min(len(x), len(lFDepth))], lFDepth[0: min(len(x), len(lFDepth))], colorsList[0])
    minLen = min(len(x), len(wpsFilterData))
    axes[1].axis('off')  # 去掉坐标轴
    axes[1].plot(x[0: minLen], wpsFilterData[0: minLen], colorsList[1])
    # plt.show()
    classFlag = 0
    plt.savefig("/mnt/X500/farmers/chenlb/WpsImage/synder_image/Wpsimage_range/" + str(classFlag) +"_"+ str (picIndex) +"_" + str(start) +"_"+ str(end) + "_Chr"+contig+"_Ocr.jpg")
    # plt.show()
    plt.close()
    # plt.savefig("/home/chenlb/MyWpsPro2/images/train/testNegative/" + str(0) +"_"+ str (picIndex) + "_HK_TSS.jpg")
    # file = open("/mnt/X500/farmers/chenlb/WpsImage/synder_image/Wpsimage_range/" + str(classFlag) +"_"+ str (picIndex) +"_" + str(start) +"_"+ str(end) + "_Chr"+contig+"_Ocr.txt", "w+")
    # file = open("/home/chenlb/MyWpsPro2/images/train/testNegative/" + str(0) +"_"+ str (picIndex) + "_HK_TSS.txt", "w+")
    # file.write(str(classFlag))
    # file.close()
    # print(str(1) +"_"+ str (picIndex) + "_HK_TSS.jpg")


def drawPeaksWithDepthByFftForClassify(lFdepth_Nor, smoothWpsList_Nor, picIndex, classFlag):
    t = np.array([i for i in range(0, len(smoothWpsList_Nor))])
    lFdepth_NorFft_t_filtered, lFdepth_NorMask = fftFilter(lFdepth_Nor, None, 0.015)
    fig, axes = plt.subplots(3, 1, figsize=(4, 4))
    axes[0].axis('off')  # 去掉坐标轴
    axes[0].plot(t, lFdepth_Nor, color="blue")
    axes[1].axis('off')  # 去掉坐标轴
    axes[1].plot(t, lFdepth_NorFft_t_filtered.real, color="red")
    axes[2].axis('off')  # 去掉坐标轴
    axes[2].plot(t, smoothWpsList_Nor, color="blue")
    plt.savefig("/mnt/X500/farmers/chenlb/WpsImage/synder_image/Wpsimage_range2/" + str(classFlag) + "_" + str(
        picIndex) + "_" + str(start) + "_" + str(end) + "_Chr" + contig + "_Ocr.jpg")
    picIndex+= 1
    plt.close()

def getTheta(x1, y1): #由(x2,y2)逆时针旋转到(x1,y1)
    theta = np.arctan2(y1, x1)
    if theta < 0:
        theta = 2 * math.pi + theta
    return np.degrees(theta)

def getMidVector(modelList): #获得两个向量的和的弧度
    '''
    :param modelList:
    :return:
    '''
    k1 = modelList[0].coef_
    k2 = modelList[1].coef_
    b1 = modelList[0].intercept_
    b2 = modelList[1].intercept_
    intersectionX = (b2 - b1) / (k1 - k2)  # 交点
    intersectionY = (k2 * b1 - k1 * b2) / (k2 - k1)
    #
    point1X = 0
    point1Y = b1
    point2X = 1990
    point2Y = k2 * point2X + b2

    vectorX = point1X - intersectionX + point2X - intersectionX
    vectorY = point1Y - intersectionY + point2Y - intersectionY

    radian = getTheta(vectorX, vectorY)
    radian1 = getTheta(point1X - intersectionX, point1Y - intersectionY)
    radian2 = getTheta(point2X - intersectionX, point2Y - intersectionY)
    if abs(radian1 - radian2) >= 180:
        radianMid = (radian1 + radian2) / 2 + 180
    else:
        radianMid = (radian1 + radian2) / 2
    radianMid %= 360

    return [radianMid, radian[0][0], intersectionX[0][0], intersectionY[0][0], vectorX[0][0], vectorY[0][0]]

def drawLinearLine2(modelList, xList, yList, title):
    '''
    绘制涉及的向量以及其角平分线
    :param modelList:
    :param xList:
    :param yList:
    :param title:
    :return:
    '''
    k1 = modelList[0].coef_[0][0]
    k2 = modelList[1].coef_[0][0]
    b1 = modelList[0].intercept_[0]
    b2 = modelList[1].intercept_[0]
    print(k1, b1)
    pointX = (b2 - b1) / (k1 - k2) #交点
    pointY = (k2 * b1 - k1 * b2) / (k2 - k1)

    point1X = 0
    point1Y = b1
    point2X = 1990
    point2Y = k2 * point2X + b2

    point3X = point1X - pointX + point2X - pointX
    point3Y = point1Y - pointY + point2Y - pointY

    vec = getMidVector(modelList)
    ang = vec[0][0][0]
    kMid = math.tan(math.radians(ang))
    interceptMid = pointY - kMid * pointX
    print([ang, kMid, interceptMid])

    figure, axes = plt.subplots(2, 1, figsize=(8, 9))

    if (ang <= 180 and ang >= 0):
        axes[0].plot([pointX, ((pointY + 10) - interceptMid) / kMid], [pointY, pointY + 10], color='k')
    else:
        axes[0].plot([pointX, ((pointY - 10) - interceptMid) / kMid], [pointY, pointY - 10], color='k')

    axes[0].plot(xList[2], yList[2], color='k')
    axes[0].plot(xList[0], modelList[0].coef_ * xList[0] + modelList[0].intercept_, color='r')
    axes[0].plot(xList[1], yList[1], color='y')
    axes[0].plot(xList[1], modelList[1].coef_ * xList[1] + modelList[1].intercept_, color='g')
    axes[0].plot(xList[2], modelList[1].coef_ * xList[2] + modelList[2].intercept_, color='b')
    axes[0].scatter(pointX, pointY, marker='x')
    axes[0].scatter(point1X, point1Y, marker='x')
    axes[0].scatter(point2X, point2Y, marker='x')
    axes[0].scatter(pointX + point3X, pointY + point3Y, marker='x')

    axes[0].plot([point1X, pointX + point3X], [point1Y, pointY + point3Y], color='k')
    axes[0].plot([point2X, pointX + point3X], [point2Y, pointY + point3Y], color='k')
    axes[0].plot([pointX, pointX + point3X], [pointY, pointY + point3Y], color='k')
    axes[1].plot([i for i in range(len(smoothWpsList_Nor))], smoothWpsList_Nor)
    axes[0].set_title(title)
    plt.savefig('/mnt/X500/farmers/chenlb/WpsImage/panel/linear' + str(picIndex) + '.jpg')

    plt.close()

def fftFilter(data, lowFreq, highFreq):
    B = 10.0
    Fs = 2 * B
    delta_f = 1
    N = len(data)
    T = N / Fs
    t = np.array([i for i in range(0, len(data))])
    dataFft = fftpack.fft(data)
    dataFreq = fftpack.fftfreq(n=len(data), d=1 / Fs)
    mask = np.where(dataFreq >= 0)
    if lowFreq != None:
        dataFft = dataFft * (abs(dataFreq) > lowFreq)
    if highFreq != None:
        dataFft = dataFft * (abs(dataFreq) < highFreq)
    dataFiltered = dataFft
    dataFftTFiltered = fftpack.ifft(dataFiltered)
    return dataFftTFiltered, mask


if __name__ == '__main__':
    # wpsFilePath = '/mnt/X500/farmers/chenhx/02.project/hexiaoti/05.NDR_chr12/cellNornal.bam.list.result.wps.xls'
    # wpsFilePath = '/home/chenlb/WPSCalProject/ndr.txt'
    # wpsList = []
    # dataList = getBedtoolsWpsData('/mnt/X500/farmers/chenlb/CellData/IH01/1.txt', 100000)
    # plt.plot(dataList[35000:36000,1], dataList[35000:36000,2])
    # plt.show()

    NormalPathList = ['/mnt/X500/farmers/chenhx/02.project/hexiaoti/02.cellPaper_data/IH01.bam',
                      '/mnt/X500/farmers/chenhx/02.project/hexiaoti/02.cellPaper_data/BH01.bam',
                      '/mnt/X500/farmers/chenhx/02.project/hexiaoti/02.cellPaper_data/IH02.bam']

    HkLiverPathList = ['/mnt/X500/farmers/chenlb/cfDNA_HK/T99.sorted.bam',
                       '/mnt/X500/farmers/chenlb/cfDNA_HK/TBR1576/TBR1576.sorted.bam',
                       '/mnt/X500/farmers/chenlb/cfDNA_HK/TBR1453/TBR1453.sorted.bam',
                       '/mnt/X500/farmers/chenlb/cfDNA_HK/TBR1450/TBR1450.sorted.bam',
                       '/mnt/X500/farmers/chenlb/cfDNA_HK/TBR1574/TBR1574.sorted.bam',
                       '/mnt/X500/farmers/chenlb/cfDNA_HK/TBR1575/TBR1575.sorted.bam',
                       '/mnt/X500/farmers/chenlb/cfDNA_HK/TBR1449/TBR1449.sorted.bam',
                       '/mnt/X500/farmers/chenlb/cfDNA_HK/TBR1454/TBR1454.sorted.bam',
                       '/mnt/X500/farmers/chenlb/cfDNA_HK/TBR1451/TBR1451.sorted.bam',
                       '/mnt/X500/farmers/chenlb/cfDNA_HK/TBR1452/TBR1452.sorted.bam',
                       '/mnt/X500/farmers/chenlb/cfDNA_HK/TBR1448/TBR1448.sorted.bam',
                       '/mnt/X500/farmers/chenlb/cfDNA_HK/TBR1577/TBR1577.sorted.bam',
                       '/mnt/X500/farmers/chenlb/cfDNA_HK/TBR1573/TBR1573.sorted.bam',
                       '/mnt/X500/farmers/chenlb/cfDNA_HK/TBR1578/TBR1578.sorted.bam'
                       ]

    HKColorectalCancerPathList = [
        '/mnt/X500/farmers/chenlb/cfDNA_HK/colorectal_cancer/TBR901/TBR901.sorted.bam',
        '/mnt/X500/farmers/chenlb/cfDNA_HK/colorectal_cancer/TBR910/TBR910.sorted.bam',
        '/mnt/X500/farmers/chenlb/cfDNA_HK/colorectal_cancer/TBR910/TBR910.sorted.bam',
        '/mnt/X500/farmers/chenlb/cfDNA_HK/colorectal_cancer/TBR911_rep1/TBR911.rep1.sorted.bam',
        '/mnt/X500/farmers/chenlb/cfDNA_HK/colorectal_cancer/TBR911_rep2/TBR911.rep2.sorted.bam',
        '/mnt/X500/farmers/chenlb/cfDNA_HK/colorectal_cancer/TBR912/TBR912.sorted.bam',
        '/mnt/X500/farmers/chenlb/cfDNA_HK/colorectal_cancer/TBR914_rep1/TBR914.rep1.sorted.bam',
        '/mnt/X500/farmers/chenlb/cfDNA_HK/colorectal_cancer/TBR914_rep2/TBR914.rep2.sorted.bam',
        '/mnt/X500/farmers/chenlb/cfDNA_HK/colorectal_cancer/TBR916_rep1/TBR916.rep1.sorted.bam',
        '/mnt/X500/farmers/chenlb/cfDNA_HK/colorectal_cancer/TBR916_rep2/TBR916.rep2.sorted.bam'
    ]
    # NormalPathList = [
    # '/mnt/GenePlus001/prod/workspace/IFA20200621001/OncoS-BColon_D4uid/output/200008369B1CD_200008369B3P1D/cancer/5_recal_bam/200008369B1CD_200008369B3P1D_cancer_sort_markdup_realign_recal.bam',
    # '/mnt/GenePlus001/prod/workspace/IFA20200621001/OncoS-BColon_D4uid/output/200008369B1CD_200008369B3P1D/normal/5_recal_bam/200008369B1CD_200008369B3P1D_normal_sort_markdup_realign_recal.bam',
    # '/mnt/X500/farmers/haoshg/bnc/cap_20200612/V100001021/output/170016793BCD_170016793BPD_3/cancer/5_recal_bam/170016793BCD_170016793BPD_3_cancer_sort_markdup_realign_recal.bam'
    # ,'/mnt/GenePlus001/prod/workspace/IFA20200620002/OncoD_D4uid/output/200004211BCD_209002355BL1s1D/cancer/5_recal_bam/200004211BCD_209002355BL1s1D_cancer_sort_markdup_realign_recal.bam'
    # ,'/mnt/GenePlus001/prod/workspace/IFA20200620002/OncoD_D4uid/output/208010407BCD_208010407BP1D/cancer/5_recal_bam/208010407BCD_208010407BP1D_cancer_sort_markdup_realign_recal.bam'
    # ,'/mnt/GenePlus001/prod/workspace/IFA20200620002/OncoD_D4uid/output/208011508BCD_208011508BP1D/cancer/5_recal_bam/208011508BCD_208011508BP1D_cancer_sort_markdup_realign_recal.bam'
    # ,'/mnt/GenePlus001/prod/workspace/IFA20200620002/OncoD_D4uid/output/200008879BCD_200008879BP1D/cancer/5_recal_bam/200008879BCD_200008879BP1D_cancer_sort_markdup_realign_recal.bam'
    # ,'/mnt/GenePlus001/prod/workspace/IFA20200620002/OncoMD_D4uid/output/200020177B1CD_200020177B1P1D/cancer/5_recal_bam/200020177B1CD_200020177B1P1D_cancer_sort_markdup_realign_recal.bam'
    # ,'/mnt/GenePlus001/prod/workspace/IFA20200620002/OncoD_D4uid/output/200009014BCD_200009014BP1D/cancer/5_recal_bam/200009014BCD_200009014BP1D_cancer_sort_markdup_realign_recal.bam'
    # ,'/mnt/GenePlus001/prod/workspace/IFA20200620002/OncoD_D4uid/output/200021859BCD_200021859BP1D/cancer/5_recal_bam/200021859BCD_200021859BP1D_cancer_sort_markdup_realign_recal.bam'
    # ,'/mnt/GenePlus001/prod/workspace/IFA20200621001/OncoS-BColon_D4uid/output/200008369B1CD_200008369B3P1D/cancer/5_recal_bam/200008369B1CD_200008369B3P1D_cancer_sort_markdup_realign_recal.bam'
    # ]
    mergeBamListPath = '/home/kuangni/chenhx/chenlb/work/result/merge.bam.list'
    lungBamListPath = '/home/kuangni/chenhx/chenlb/work/result/Lung.bam.list.new'
    OvarianBamListPath = '/home/kuangni/chenhx/chenlb/work/result/Ovarian.bam.list'
    panelDataListPath = '/home/chenlb/MyWpsPro2/panle_bamlist.list'
    # bamfileList = getPathList(NormalPathList)
    bamfileList = HkLiverPathList
    # print(bamfileList)
    # bamfileList = NormalPathList

    # pointFilePath = '/mnt/X500/farmers/chenhx/02.project/hexiaoti/05.NDR_all/wholegenome.20k.filter.bin'
    # pointFilePath = '/home/chenlb/MyWpsPro2/OverLap.OCR.HK.chr1.bed'
    pointFilePath = '/home/chenlb/result/HK.all.txt.bed'
    # pointFilePath = '/home/chenlb/MyWpsPro2/OCRS_1029_chr1_1.hk.txt'
    # pointFilePath = '/home/chenlb/MyWpsPro2/OverLap.HK.chr1.bed'
    # pointFilePath = '/home/chenlb/MyWpsPro2/NoOverLap.HK.chr1.bed'
    # pointFilePath = '/home/chenlb/result/Overlap.ndr_HK.bed'
    # pointFilePath = '/home/kuangni/chenhx/chenlb/work/result/TSS.bed'
    # pointFilePath = '/home/kuangni/chenhx/chenlb/work/result/wholegenome.20k.filter.bin'
    # pointFilePath = '/home/kuangni/chenhx/chenlb/work/result/Overlap.HK_ndr.2000.txt.bed'
    # pointFilePath = '/home/kuangni/chenhx/chenlb/work/result/HK_gene_info.chr1_5.txt.bed.sort'
    # pointFilePath = '/home/chenlb/WPSCalProject/filePathList/HK_gene_info.chr1_5.txt.bed.sort'
    # pointFilePath = '/home/chenlb/result/OCRs/NoOverlapWith_ATAC_DNase_HKTSS.bed'
    # geneDict = getTSSPoint('/home/chenlb/WPSCalProject/filePathList/GRCh37.gene.bed', 100000)

    #'/home/chenlb/MyWpsPro2/Tissue-Specific.chr1.bed'
    pointFilePathList = ['/home/chenlb/result/HK.all.txt.bed',
                         '/mnt/X500/farmers/chenhx/02.project/hexiaoti/05.NDR_all/wholegenome.20k.filter.bin',
                         '/home/chenlb/result/HK.all.txt.bed',
                         '/home/chenlb/result/HK.all.txt.bed']
    #                        '/home/chenlb/MyWpsPro2/OCRS_0908_1_1.txt',
    # pointList = getPointData(pointFilePath, 1000000000)
    # allPoint = len(pointList)
    # allPoint = len(pointList)
    round = 0
    s = e = 0
    peaksList = []
    # kal_filter = Kalman_filter.kalman_filter(Q=0.1, R=10)
    # chr12 34484000 34561000
    t1 = time.clock()
    picIndex = 0

    peakWdith = []
    peakDis = []
    peakDisSet = set()
    # for geneName in open('/home/chenlb/WPSCalProject/filePathList/HK_gene_names.txt'):
    # if not geneDict.__contains__(geneName[:-1]):
    #     continue
    areaList = []
    nonePeakAreaList = []

    peakObjects = []
    peakCnt = []
    depthList = []
    bigDepthList = []
    smallDepthList = []
    heightList = []
    widthList = []
    leftKList = []
    rightKList = []
    angelList = []
    areaVarList = []
    heightVarList = []
    widthVarList = []
    leftKVarList = []
    rightKVarList = []
    angelVarList = []
    areaVarList = []
    vectorList = []
    coefList = []
    interceptList = []



    modelList = []
    xList = []
    yList = []
    startLiist = []
    endList = []
    contigList = []
    # with open('rf.pickle', 'rb') as fr:
    #     rfModel = pickle.load(fr)
    for i in range(2):
        # if (i >= 1):
        #     break
        pointList = getPointData(pointFilePathList[i], 1000000000)

        peakObjectsTemp = []
        pCnt = []
        depths = []
        bigDepths = []
        smallDepths = []
        height = []
        width = []
        leftK = []
        rightK = []
        angel = []
        area = []
        heightVar = []
        widthVar = []
        leftKVar = []
        rightKVar = []
        angelVar = []
        areaVar = []
        round = 0
        modelsS = []
        vectorsS = []
        startS = []
        endS = []
        contigS = []
        for point in pointList:
            # print('*************************************  round : ', round, '  ->  ', allPoint,
            #       ' *************************************')
            round += 1
            if round < 0:
                continue;
            if round > 10000:
                break
            contig = point[0]
            if i < 10:
                start = int((int(point[1]) + int(point[2])) / 2) - 1000
                # start = int(point[1])
                # start = int(point[1]) + randint(300, 1000)
                # end = int(point[2]) + randint(0, 100)
                end = start + 2000 + randint(-1000, 500)
                end = start + 2000
            elif i == 2:  # 左移
                start = int((int(point[1]) + int(point[2])) / 2) + randint(-200, 200)
                end = start + 2000 + randint(-1000, 500)
                end = start + 2000
            elif i == 3:  # 左移
                end = int((int(point[1]) + int(point[2])) / 2) + randint(-200, 200)
                start = end - 2000 + randint(-1000, 500)
                start = end - 2000
            # end = int(point[2])
            step = end - start
            peaksList = []
            dataList = []
            x = np.arange(start, end)
            length = step + 1
            print(length)
            wpsList_Nor, lFdepth_Nor, sFdepth_Nor = callOneBed(bamfileList, 'chr' + str(contig), start, end, win=120)
            # wpsList_Nor, lFdepth_Nor, sFdepth_Nor = callOneBed(bamfileList, contig, start, end, win=120)
            raw_lFdepth_Nor = np.array(lFdepth_Nor)
            rawWPS = np.array(wpsList_Nor)
            adjustWpsList_Nor = AdjustWPS(wpsList_Nor)
            # adjustWpsList_Nor = lFdepth_Nor
            try:
                base = peakutils.baseline(adjustWpsList_Nor, 8)
            except (ZeroDivisionError, ValueError) as reason:  # 'ZeroDivisionError'除数等于0的报错方式^M
                base = np.zeros(len(adjustWpsList_Nor))
            # base = baseline_als(adjustWpsList_Nor, 1, 1, 10)


            try:
                adjustWpsList_Nor = np.subtract(adjustWpsList_Nor, base)
                smoothWpsList_Nor = savgol_filter_func(adjustWpsList_Nor, 51, 1)
                smoothWpsList_Nor = preprocessing.minmax_scale(smoothWpsList_Nor)
            except ValueError:
                continue
            lFdepth_Nor = savgol_filter_func(lFdepth_Nor, 801, 2)

            # plt.plot([i for i in range(len(sFdepth_Nor))], preprocessing.minmax_scale(np.array(sFdepth_Nor)) + 2)
            # plt.plot([i for i in range(len(lFdepth_Nor))], preprocessing.minmax_scale(np.array(lFdepth_Nor)) + 1)
            # plt.plot([i for i in range(len(smoothWpsList_Nor))], smoothWpsList_Nor)
            # plt.savefig('/mnt/X500/farmers/chenlb/WpsImage/panel/depth_' + str(picIndex) + '.jpg')
            # plt.close()
            # picIndex += 1
            #


            # plt.plot([i for i in range(len(smoothWpsList_Nor))], smoothWpsList_Nor)
            # plt.plot([i for i in range(len(smoothWpsList_Nor))], adjustWpsList_Nor)
            # plt.savefig(
            #     "/mnt/X500/farmers/chenlb/WpsImage/panel/hk_images/" + str(picIndex) + "_" + str(start) + "_" + str(
            #         end) + "_Chr" + contig + "_Ocr.jpg")
            # plt.close()

            peakHeight = []
            ###
            for data in [smoothWpsList_Nor]:
                peaks = scipy_signal_find_peaks(data, height=0.28, distance=25, prominence=0.25, width=[25, 170])
                peakObjectList = getValley(data, adjustWpsList_Nor, peaks[1], 5)
                # peakObjectListCWT = getValley(adjustWpsList, rawDataList, peaksCWT[1], 4)

                # mergeValley(peakObjectList)
                # mergeValley(peakObjectListCWT)
                # peakObjectList = mergePeaks(peakObjectList)
                peaksList.append([adjustWpsList_Nor, peaks[1], peakObjectList])
                peaksList.append([data, peaks[1], peakObjectList])
            try:
                getALLPeaksAveHeight(peakObjectList=peakObjectList, normaliedRawDataList=smoothWpsList_Nor,
                                     nonePeakAreaList=nonePeakAreaList)
            except:
                continue

            sublen = int(len(lFdepth_Nor) / 3)

            models = []

            for index in range(2):
                model, new_x, new_y = linearJudgeNDR(lFdepth_Nor, lFdepth_Nor, index *sublen, (index + 2) * sublen - 1, picIndex, False)
                picIndex += 1
                models.append(model)
            model, new_x, new_y = linearJudgeNDR(lFdepth_Nor, lFdepth_Nor, 0, len(lFdepth_Nor) - 1, picIndex, False)
            picIndex += 1

            models.append(model)

            vectors = getMidVector(models)
            picIndex += 1
            # print(i,[models[0].coef_ , models[1].coef_ , models[2].coef_ ])
            vectorsS.append(vectors)
            modelsS.append(models)

            for peak in peakObjectList:
                peakObjectsTemp.append(peak)
            meanWidth, meanHeight, meanAngel, meanArea, varWidth, varHeight, varAngel, varArea = haveNearContinuouslyPeak(smoothWpsList_Nor, adjustWpsList_Nor, peakObjectList)
            #
            pCnt.append(len(peakObjectList))
            depths.append(np.sum(lFdepth_Nor) / length)
            height.append(meanHeight)
            width.append(meanWidth)
            angel.append(meanAngel)
            area.append(meanArea)
            leftK.append(np.mean([p.leftK for p in peakObjectList]))
            rightK.append(np.mean([p.rightK for p in peakObjectList]))

            heightVar.append(varHeight)
            widthVar.append(varWidth)
            angelVar.append(varAngel)
            areaVar.append(varArea)
            leftKVar.append(np.var([p.leftK for p in peakObjectList]))
            rightKVar.append(np.var([p.rightK for p in peakObjectList]))
            ndrAreaDepth, smallNdrAreaDepth, minIndex = judgeLowDepth(lFdepth_Nor, 0, len(lFdepth_Nor) - 1)
            bigDepths.append(ndrAreaDepth)
            smallDepths.append(smallNdrAreaDepth)
            startS.append(start)
            endS.append(end)
            contigS.append(int(contig))
            drawPeaksWithDepthByFftForClassify(raw_lFdepth_Nor, smoothWpsList_Nor, picIndex, i)
            picIndex += 1



            #画角平分线
            # m0 = models[0]
            # m1 = models[1]
            # if m0.coef_ <= 0 and m1.coef_ >= 0:
            #     coefOneHot = 1.0
            # elif m0.coef_ <= 0 and m1.coef_ <= 0:
            #     coefOneHot = 2.0
            # elif m0.coef_ >= 0 and m1.coef_ >= 0:
            #     coefOneHot = 3.0
            # else:
            #     coefOneHot = 4.0


            # data = np.array([[np.sum(lFdepth_Nor) / length, ndrAreaDepth, smallNdrAreaDepth, meanHeight, meanWidth, meanAngel, meanArea, leftK[-1],
            #                  rightK[-1], varHeight, varWidth, varAngel, varArea,  leftKVar[-1], rightKVar[-1], coefOneHot, models[0].coef_, models[1].coef_, models[2].coef_,
            #                  models[0].intercept_, models[1].intercept_, models[2].intercept_, vectors[0], vectors[1], vectors[2], vectors[3], vectors[4], vectors[5]]])
            # preLabel = rfModel.predict(data[:, 0:-5])
            # if preLabel[0] == i:
            #     print(int(preLabel[0]), ' == ', i ,' : True')
            # else:
            #     drawLinearLine2(models, xs, ys, str(i) + ' : ' + str(int(preLabel[0])))
            #     picIndex += 1
            # continue
            #------


            # drawPeaksWithDepth2(smoothWpsList_Nor, base, peaksList, 120, x, start,
            #                    ['长片段深度', 'sg滤波处理WPS', '原始WPS', '卷积平滑', 'sg滤波', '小波滤波'])
            # picIndex = picIndex + 1

            # drawLinearLine(models, xList, yList)



        peakObjects.append(peakObjectsTemp)
        peakCnt.append(pCnt)
        depthList.append(depths)
        bigDepthList.append(bigDepths)
        smallDepthList.append(smallDepths)
        heightList.append(height)
        widthList.append(width)
        angelList.append(angel)
        areaList.append(area)
        leftKList.append(leftK)
        rightKList.append(rightK)
        heightVarList.append(heightVar)
        widthVarList.append(widthVar)
        angelVarList.append(angelVar)
        areaVarList.append(areaVar)
        leftKVarList.append(leftKVar)
        rightKVarList.append(rightKVar)
        modelList.append(modelsS)
        vectorList.append(vectorsS)
        startLiist.append(startS)
        endList.append(endS)
        contigList.append(contigS)
    # std = MinMaxScaler()
    # std = std.fit(vectorsS)
    # vectorsS = std.transform(vectorsS)
    # vec = pd.DataFrame(vectorsS)
    # vec = vec.dropna(axis=0, how='any')
    # print(vectorsS)
    # sns.heatmap(vec)
    # plt.savefig('/mnt/X500/farmers/chenlb/WpsImage/panel/vec' + str(picIndex) + '.jpg')

    # drawKde3(dataList = vectorList, labelList=['OCRs_wave_vector', 'non_OCRs_wave_vector', 'OCRs_class3_wave_vector', 'OCRs_class3_wave_vector'],
    #         savepath='/mnt/X500/farmers/chenlb/WpsImage/panel/', picIndex = picIndex)
    #
    # drawKde3(dataList = modelList, labelList=['OCRs_wave_coef', 'non_OCRs_wave_coef', 'OCRs_class3_wave_coef', 'OCRs_class3_wave_coef'],
    #         savepath='/mnt/X500/farmers/chenlb/WpsImage/panel/', picIndex = picIndex)
    # picIndex += 1
    # drawKde(dataList = areaList, labelList=['OCRs_peak_area', 'non_OCRs_peak_area', 'OCRs_class3_peak_area', 'OCRs_class3_peak_area'],
    #         savepath='/mnt/X500/farmers/chenlb/WpsImage/panel/peak_area_mean.jpg')
    #
    # drawKde(data=widthList, labelList= ['OCRs_peak_width', 'non_OCRs_peak_width', 'OCRs_class3_peak_width', 'OCRs_class3_peak_width'],
    #         savepath='/mnt/X500/farmers/chenlb/WpsImage/panel/peak_width_mean.jpg')
    #
    # drawKde(dataList=angelList,
    #         labelList=['OCRs_peak_angel', 'non_OCRs_peak_angel', 'OCRs_class3_peak_angel', 'OCRs_class3_peak_angel'],
    #         savepath='/mnt/X500/farmers/chenlb/WpsImage/panel/peak_angel_mean.jpg')
    # #
    # drawKde(dataList=heightList,
    #         labelList=['OCRs_peak_height', 'non_OCRs_peak_height', 'OCRs_class3_peak_height', 'OCRs_class3_peak_height'],
    #         savepath='/mnt/X500/farmers/chenlb/WpsImage/panel/peak_height_mean.jpg')

    # drawKde(dataList = areaList, labelList=['OCRs_peak_area_var', 'non_OCRs_peak_area_var', 'OCRs_class3_peak_area_var', 'OCRs_class3_peak_area_var'],
    #         savepath='/mnt/X500/farmers/chenlb/WpsImage/panel/peak_area_var.jpg')
    #
    # drawKde(data=widthList, labelList= ['OCRs_peak_width_var', 'non_OCRs_peak_width_var', 'OCRs_class3_peak_width_var', 'OCRs_class3_peak_width_var'],
    #         savepath='/mnt/X500/farmers/chenlb/WpsImage/panel/peak_width_var.jpg')
    #
    # drawKde(data=angelList,
    #         labelList=['OCRs_peak_angel_var', 'non_OCRs_peak_angel_var', 'OCRs_class3_peak_angel_var', 'OCRs_class3_peak_angel_var'],
    #         savepath='/mnt/X500/farmers/chenlb/WpsImage/panel/peak_angel_var.jpg')
    #
    # drawKde(data=heightList,
    #         labelList=['OCRs_peak_height_var', 'non_OCRs_peak_height_var', 'OCRs_class3_peak_height_var', 'OCRs_class3_peak_height_var'],
    #         savepath='/mnt/X500/farmers/chenlb/WpsImage/panel/peak_height_var.jpg')


    # drawKde(data1=np.array([lk for lk in leftKList[0]]),
    #         data2=np.array([lk for lk in leftKList[1]]), data3=np.array([lk for lk in leftKList[2]]),
    #         data4=np.array([lk for lk in leftKList[3]]), label1='OCRs_peak_leftK',
    #         label2='non_OCRs_peak_leftK', label3='OCRs_class3_peak_leftK', label4='OCRs_class3_peak_leftK',
    #         savepath='/mnt/X500/farmers/chenlb/WpsImage/panel/peak_leftK_std.jpg')
    #
    # drawKde(data1=np.array([rk for rk in rightKList[0]]),
    #         data2=np.array([rk for rk in rightKList[1]]), data3=np.array([rk for rk in rightKList[2]]),
    #         data4=np.array([rk for rk in rightKList[3]]), label1='OCRs_peak_rightK',
    #         label2='non_OCRs_peak_rightK', label3='OCRs_class3_peak_rightK', label4='OCRs_class3_peak_rightK',
    #         savepath='/mnt/X500/farmers/chenlb/WpsImage/panel/peak_rightK_std.jpg')


    # drawKde(data1=np.array([h for h in heightList[0]]),
    #         data2=np.array([h for h in heightList[1]]), data3=np.array([h for h in heightList[2]]),
    #         data4=np.array([h for h in heightList[3]]), label1='OCRs_peak_area',
    #         label2='non_OCRs_peak_area', label3='OCRs_class3_peak_area', label4='OCRs_class3_peak_area',
    #         savepath='/mnt/X500/farmers/chenlb/WpsImage/panel/peak_area_std.jpg')
    #
    # drawKde(data1=np.array([w for w in widthList[0]]),
    #         data2=np.array([w for w in widthList[1]]), data3=np.array([w for w in widthList[2]]),
    #         data4=np.array([w for w in widthList[3]]), label1='OCRs_peak_width',
    #         label2='non_OCRs_peak_width', label3='OCRs_class3_peak_width', label4='OCRs_class3_peak_width',
    #         savepath='/mnt/X500/farmers/chenlb/WpsImage/panel/peak_width_std.jpg')
    #
    # drawKde(data1=np.array([lk for lk in leftKList[0]]),
    #         data2=np.array([lk for lk in leftKList[1]]), data3=np.array([lk for lk in leftKList[2]]),
    #         data4=np.array([lk for lk in leftKList[3]]), label1='OCRs_peak_leftK',
    #         label2='non_OCRs_peak_leftK', label3='OCRs_class3_peak_leftK', label4='OCRs_class3_peak_leftK',
    #         savepath='/mnt/X500/farmers/chenlb/WpsImage/panel/peak_leftK_std.jpg')
    #
    # drawKde(data1=np.array([rk for rk in rightKList[0]]),
    #         data2=np.array([rk for rk in rightKList[1]]), data3=np.array([rk for rk in rightKList[2]]),
    #         data4=np.array([rk for rk in rightKList[3]]), label1='OCRs_peak_rightK',
    #         label2='non_OCRs_peak_rightK', label3='OCRs_class3_peak_rightK', label4='OCRs_class3_peak_rightK',
    #         savepath='/mnt/X500/farmers/chenlb/WpsImage/panel/peak_rightK_std.jpg')

    # drawKde(data1=np.array([pc for pc in peakCnt[0]]),
    #         data2=np.array([pc for pc in peakCnt[1]]), data3=np.array([pc for pc in peakCnt[2]]),
    #         data4=np.array([pc for pc in peakCnt[3]]), label1='OCRs_peak_cnt',
    #         label2='non_OCRs_peak_cnt', label3='OCRs_class3_peak_cnt', label4='OCRs_class3_peak_cnt',
    #         savepath='/mnt/X500/farmers/chenlb/WpsImage/panel/peak_cnt.jpg')


    #
    # drawKde(data1=np.array([bd for bd in bigDepthList[0]]),
    #         data2=np.array([bd for bd in bigDepthList[1]]), data3=np.array([bd for bd in bigDepthList[2]]),
    #         data4=np.array([bd for bd in bigDepthList[3]]), label1='OCRs_big_area_depth',
    #         label2='non_OCRs_big_area_depth', label3='OCRs_class3_big_area_depth', label4='OCRs_class3_big_area_depth',
    #         savepath='/mnt/X500/farmers/chenlb/WpsImage/panel/big_area_depth.jpg')
    #
    # drawKde(data1=np.array([sd for sd in smallDepthList[0]]),
    #         data2=np.array([sd for sd in smallDepthList[1]]), data3=np.array([sd for sd in smallDepthList[2]]),
    #         data4=np.array([sd for sd in smallDepthList[3]]), label1='OCRs_small_area_depth',
    #         label2='non_OCRs_small_area_depth', label3='OCRs_class3_small_area_depth', label4='OCRs_class3_small_area_depth',
    #         savepath='/mnt/X500/farmers/chenlb/WpsImage/panel/small_area_depth.jpg')

    #
    dataArr = []
    # print(depthList[0][0])
    # print(depthList[0][1])
    # print(depthList[0][2])
    # print('end')
    for i in range(2):
        # if i > 0:
        #     break
        for j in range(len(areaVarList[i])):
            list = []
            list.append(i)
            list.append(depthList[i][j])
            list.append(bigDepthList[i][j])
            list.append(smallDepthList[i][j])
            list.append(heightList[i][j])
            list.append(widthList[i][j])
            list.append(angelList[i][j])
            list.append(areaList[i][j])
            list.append(leftKList[i][j])
            list.append(rightKList[i][j])
            list.append(heightVarList[i][j])
            list.append(widthVarList[i][j])
            list.append(angelVarList[i][j])
            list.append(areaVarList[i][j])
            list.append(leftKVarList[i][j])
            list.append(rightKVarList[i][j])
            m0 = modelList[i][j][0]
            m1 = modelList[i][j][1]
            if m0.coef_ <= 0 and m1.coef_ >= 0:
                list.append(1.0)
            elif m0.coef_ <= 0 and m1.coef_ <= 0:
                list.append(2.0)
            elif m0.coef_ >= 0 and m1.coef_ >= 0:
                list.append(3.0)
            else :
                list.append(4.0)
            list.append(modelList[i][j][0].coef_)
            list.append(modelList[i][j][1].coef_)
            list.append(modelList[i][j][2].coef_)
            list.append(modelList[i][j][0].intercept_)
            list.append(modelList[i][j][1].intercept_)
            list.append(modelList[i][j][2].intercept_)
            for v in vectorList[i][j]:
                list.append(v)
            list.append(contigList[i][j])
            list.append(startLiist[i][j])
            list.append(endList[i][j])
            dataArr.append(list)


        # rfModel.predict(X[0:1])
    dataArr = np.array(dataArr)
    # print(len(dataArr))
    # score = rfModel.score(dataArr[:,1:-5],dataArr[:,0])
    # print('K-Fold acc = ', score, ' mean acc = ', np.mean(score))
    df = pd.DataFrame(np.array(dataArr))
    # df.columns = ['label', 'depth', 'bigAreaDepth', 'smallDepth', 'height', 'width']
    df.to_csv('./dataAll_DHS_HKLiver_train.csv', sep=',', header=True, index=True)



    # peaksList = []
    # step = 4000
    # rawData = np.array(rawWPS)
    # wpsData = np.array(smoothWpsList_Nor)
    # t = np.array([i for i in range(0,len(smoothWpsList_Nor))])
    # lFdepth_NorFft_t_filtered, lFdepth_NorMask = fftFilter(lFdepth_Nor, None, 0.015)
    # smoothWpsFft_filtered, smoothWpsMask = fftFilter(smoothWpsList_Nor, 0.05, 0.12)
    # fig, axes = plt.subplots(3, 1, figsize=(8, 8))
    # axes[0].axis('off')  # 去掉坐标轴
    # axes[0].plot(t, lFdepth_Nor, color="blue")
    # axes[1].axis('off')  # 去掉坐标轴
    # axes[1].plot(t, lFdepth_NorFft_t_filtered.real, color="red")
    # axes[2].axis('off')  # 去掉坐标轴
    # axes[2].plot(t, smoothWpsList_Nor, color="blue")
    # plt.savefig(
    #     "/mnt/X500/farmers/chenlb/WpsImage/tempImages/" + str(round) + "_" + str(start) + "_" + str(
    #         end) + "_Chr" + contig + "_Ocr.jpg")
    # '''
    #     classFlag = 1, 负样本
    #     classFlag = 2, 正样本全
    #     classFlag = 3, 正样本左
    #     classFlag = 4, 正样本右
    # '''
    # classFlag = 4
