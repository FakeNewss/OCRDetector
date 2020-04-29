from WpsCal import *
import pysam
from scipy.stats import pearsonr
from DrawTSSWps import *
import statsmodels.api as sm

def wpsAndGcCal(wpsList, bamfile, win, contig, start, end, s, e):
    for data in bamfile.fetch(contig=contig, start=start, end=end):
        # cnt += 1
        # print(data.template_length,data.reference_start,' -- ',data.query_name ,data.query_sequence)
        if data.reference_start - start >= 0 and data.reference_start - start + data.isize < len(wpsList) and data.isize > 0 and data.isize < 1000:
            listIndex = data.reference_start - start
            biWin = int(win / 2)
            for index in range(listIndex, listIndex + data.isize):
                if index >= listIndex + biWin and index <= listIndex + data.isize - biWin:
                    wpsList[index] += 1
                else:
                    wpsList[index] -= 1
    coverageArray = bamfile.count_coverage(contig=contig, start=start, end=end)
    #A C G T

    GCRatio = (np.sum(coverageArray[1]) + np.sum(coverageArray[2])) / (np.sum(coverageArray[:]))
    sum = 0
    cnt = 0
    for wps in wpsList:
        if wps != 0:
            sum += wps
            cnt += 1
    mean = sum / cnt
    print(GCRatio, ' ', mean)
    return GCRatio, mean

def writeGcMeanWpsToFile(contig, start, end, GCRatio, meanWPS):
    with open('gc_mean.txt', mode='a+') as f:
        list = [contig, start, end, GCRatio, meanWPS]
        f.write(str(list)+ '\n')

def getGCData(filepath):
    dataList = []
    with open('gc_mean.txt', mode='r') as f:
        while True:
            line = f.readline()
            if line == None or len(line) <= 1:
                break
            data = line[1:-2].split(',')
            dataList.append([data[3],data[4]])
    return np.array(dataList, dtype=np.float)


def getPointData1(pointFilePath, cnt):
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
        pointList.append(data)
    return pointList


if __name__ == '__main__':
    NormalPathList = ['/mnt/X500/farmers/chenhx/02.project/hexiaoti/02.cellPaper_data/IH01.bam',
                      '/mnt/X500/farmers/chenhx/02.project/hexiaoti/02.cellPaper_data/BH01.bam',
                      '/mnt/X500/farmers/chenhx/02.project/hexiaoti/02.cellPaper_data/IH02.bam']
    bamfileList = NormalPathList
    pointFilePath = '/home/chenlb/WPSCalProject/filePathList/TSS.info.bed'
    pointList = getPointData1(pointFilePath, 10000000)
    allPoint = len(pointList)
    round = 0
    s = e = 0
    peaksList = []
    peakWdith = []
    peakDis = []
    peakDisSet = set()
    # for geneName in open('/home/chenlb/WPSCalProject/filePathList/HK_gene_names.txt'):
    # if not geneDict.__contains__(geneName[:-1]):
    #     continue
    for point in pointList:
        print('*************************************  round : ', round, '  ->  ', allPoint,
              ' *************************************')
        round += 1
        if round > 60:
            break
        # gene = geneDict[geneName[:-1]]
        # contig = gene.chr
        # start = gene.tssBinStart
        # end = gene.tssBinEnd
        # step = end - start
        contig = point[0]
        start = int(point[1]) - 2000
        end = int(point[2]) + 2000
        step = end - start
        gene_dir = point[3]
        gene_name = point[4]
        peaksList = []
        dataList = []
        x = np.arange(start, end)
        length = step + 1
        wpsList_Nor, lFdepth_Nor, sFdepth_Nor = callOneBed(bamfileList, contig, start, end, win=120)
        rawWPS = np.array(wpsList_Nor)
        lowess = sm.nonparametric.lowess
        frac = 0.025
        x = np.arange(start, end, step = 1, dtype = int)
        z = lowess(wpsList_Nor[0 : end - start], x, frac=frac)
        wpsMedian = np.median(wpsList_Nor[0 : end - start])

        wpsListPred = np.subtract(wpsList_Nor[0 : end - start], np.array(z[:, 1])) * 6 + wpsMedian
        z2 = lowess(wpsListPred, x, frac=frac)
        zAdjust = preprocessing.minmax_scale(z2[:, 1])
        medZAdj = np.median(zAdjust)
        zAdjust = zAdjust - medZAdj
        zAdjust = smooth(zAdjust, 51)
        fig, axes = plt.subplots(4, 1, figsize=(14, 11))
        axes[0].grid(axis="y", linestyle='--')
        title0 = '染色体' + str(contig) + ' 区域 ' + str(start) + ' -> ' + str(end) + ' 正副链 : ' + str(dir) + ' gene name : ' + str(gene_name)
        axes[0].set_title(title0)
        axes[0].plot(x, lFdepth_Nor, color='k')
        axes[1].grid(axis="y", linestyle='--')
        axes[1].set_title('原始wps')
        axes[1].plot(x, wpsList_Nor[0 : end - start], color='k')
        axes[2].grid(axis="y", linestyle='--')
        axes[2].set_title('原始wps曲线和lowess(局部加权回归)处理后的WPS曲线（GC校正步骤)')
        line1 = axes[2].plot(x, wpsList_Nor[0 : end - start], color='#F4700B')
        line2 = axes[2].plot(x, z2[:, 1], color='#1512F8')

        axes[3].grid(axis="y", linestyle='--')
        axes[3].set_title('lowess(局部加权回归)处理后的WPS曲线（GC校正步骤)')
        # plt.plot(z[:,0], z[:, 1],color='r',lw=1)
        line3 = axes[3].plot(x, zAdjust, color='#1512F8')
        legend(loc='upper right')
        axes[2].legend((line1[0], line2[0]), ('原始wps曲线', 'lowess(局部加权回归)处理后的WPS曲线'), loc='upper right')
        plt.xlabel('1号染色体基因组位点')
        axes[2].set_ylabel('WPS值')
        axes[3].set_ylabel('标准化WPS值')
        plt.show()
# if __name__ == '__main__':

    # gcAndwpsMeanList = getGCData('/home/chenlb/WPSCalProject/gc_mean.txt')
    # plt.scatter(gcAndwpsMeanList[:,0], gcAndwpsMeanList[:,1], s=2, marker='.', color='b')
    # plt.show()
    # pccs = pearsonr(gcAndwpsMeanList[:, 0], gcAndwpsMeanList[:, 1])
    # print(pccs)
    # pointFilePath = '/mnt/X500/farmers/chenhx/02.project/hexiaoti/05.NDR_all/wholegenome.20k.filter.bin'
    # sWGSDataPathList = [
    #     # 前三个为小细胞肺癌，临床分期IV
    #     '/mnt/X500/farmers/chenhx/02.project/hexiaoti/02.cellPaper_data/IH01.bam',
    #     # '/mnt/X500/farmers/chenhx/02.project/hexiaoti/02.cellPaper_data/BH01.bam',
    #     # '/mnt/X500/farmers/chenhx/02.project/hexiaoti/02.cellPaper_data/IH02.bam',
    #     ]
    # pathList = sWGSDataPathList
    # pointList = getPointData(pointFilePath, 1000000000000)
    # allPoint = len(pointList)
    # round = 1
    # gcList = []
    # wpsNumList = []
    # for point in pointList:
    #     print('*************************************  round : ', round, '  ->  ', allPoint,
    #           ' *************************************')
    #     round += 1
    #     contig = str(point[0])
    #     start = point[1]
    #     end = point[2]
    #     step = point[2] - point[1] + 200
    #     peaksList = []
    #     dataList = []
    #     wpsList = np.array([0 for i in range(step + 1)])
    #     length = step + 1
    #     # wpsSegTree = SegmentTree(len(wpsList), wpsList)
    #     mean = 0
    #     for filePath in pathList:
    #         start = point[1]
    #         end = point[2]
    #         bamfile = readFileData(filePath)
    #         win = 120
    #         # wpsList = wpsCal(bamfile, win, contig, start, end, s, e)
    #         # bamfile = readFileData(filePath)
    #
    #         GCRatio, meanWPS = wpsAndGcCal(wpsList, bamfile, win, contig, start, end, 0, 0)
    #         gcList.append(GCRatio)
    #         wpsNumList.append(meanWPS)
    #         writeGcMeanWpsToFile(contig, start, end, GCRatio, meanWPS)
            # wpsList, up, down = wpsCalBySegTree(wpsSegTree, wpsList, bamfile, win, contig, start, end, s, e)
        # wpsList = np.array(wpsList, dtype=float)
    # plt.scatter(np.array(gcList), np.array(wpsNumList), marker='.', color='k')
    # plt.show()
