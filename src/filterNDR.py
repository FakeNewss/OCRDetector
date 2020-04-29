from WpsCal import *
import SegmentTree
import scipy.signal
def getPointData(pointFilePath, cnt):
    pointList = []
    for path in pointFilePath:
        print('pointList len : ', len(pointList))
        file = open(path, 'r')
        dataList = []
        while True:
            line = file.readline()
            if line == None or cnt < 0:
                break
            cnt -= 1
            data =line[0:-1].split('\t')
            if len(data) == 1:
                break
            dataList.append(data)
        for data in dataList:
            pointList.append([str(data[0]), int(data[1]), int(data[2])])
    return pointList

if __name__ == '__main__':
    sWGSDataPathList = ['/mnt/X500/farmers/chenhx/02.project/hexiaoti/02.cellPaper_data/Breast.bam',
                        '/mnt/X500/farmers/chenhx/02.project/hexiaoti/02.cellPaper_data/Pancreatic.bam',
                        '/mnt/X500/farmers/chenhx/02.project/hexiaoti/02.cellPaper_data/Lung.bam',
                        '/mnt/X500/farmers/chenhx/02.project/hexiaoti/02.cellPaper_data/BH01.bam']
    typeList = ['Breast', 'Pancreatic', 'Lung', 'Normal']
    # sWGSDataPathList = [
    #     '/mnt/X500/farmers/chenhx/02.project/hexiaoti/02.cellPaper_data/IH01.bam',
    #     '/mnt/X500/farmers/chenhx/02.project/hexiaoti/02.cellPaper_data/BH01.bam',
    #     '/mnt/X500/farmers/chenhx/02.project/hexiaoti/02.cellPaper_data/IH02.bam',
    # ]
    pointPathList = [
        '/mnt/X500/farmers/chenhx/02.project/hexiaoti/05.NDR_all_cal2/bin/Breast.filteredNDR.xls.addpeak.uniq',
        '/mnt/X500/farmers/chenhx/02.project/hexiaoti/05.NDR_all_cal2/bin/NSCLC.filteredNDR.xls.addpeak.uniq',
        '/mnt/X500/farmers/chenhx/02.project/hexiaoti/05.NDR_all_cal2/bin/Intestine.filteredNDR.xls.addpeak.uniq',
        '/mnt/X500/farmers/chenhx/02.project/hexiaoti/05.NDR_all_cal2/bin/Ovarian.filteredNDR.xls.addpeak.uniq',
        '/mnt/X500/farmers/chenhx/02.project/hexiaoti/05.NDR_all_cal2/bin/Lung.filteredNDR.xls.addpeak.uniq',
        '/mnt/X500/farmers/chenhx/02.project/hexiaoti/05.NDR_all_cal2/bin/Pancreatic.filteredNDR.xls.addpeak.uniq'
    ]
    pointList = getPointData(pointPathList, 4)
    # filePath = sWGSDataPathList[0]
    s = e = 0
    step = 2400


    kal_filter = Kalman_filter.kalman_filter(Q=0.1, R=10)

    normalization_half_y = np.zeros(int(step / 2))
    typeIndex = 0


    for point in pointList:
        fList = []
        yfList = []
        for filePath in sWGSDataPathList:
            peaksList = []
            length = step + 1
            wpsList = [0 for i in range(step + 1)]
            wpsSegTree = SegmentTree.SegmentTree(len(wpsList), wpsList)
            bamfile = readFileData(filePath)
            win = 120
            start = point[1] - 1000
            end = point[2] + 1000
            contig = point[0]
            wpsList, up, down = wpsCalBySegTree(wpsSegTree, wpsList, bamfile, win, contig, start, end, 0, 0)
            rawDataList = wpsList;
            # wpsList = preprocessing.scale(wpsList)
            wpsList = kalman_filter(kal_filter, wpsList)
            wpsList = wpsList * scipy.signal.hann(len(wpsList), sym=0)
            freqs2, yf2, freqs1, yf1, adjustWpsList = fft(wpsList, rawDataList, False)
            yf1p = 20*np.log10(np.clip(np.abs(yf1), 1e-20, 1e100))
            fList.append(freqs1)
            yfList.append(yf1)
        fig, axes = plt.subplots(len(yfList), 1, figsize=(12, 8))
        lineList = []
        for i in range(len(yfList)):
            axes[i].grid(linestyle="--")
            line = axes[i].plot(fList[i][1:200], yfList[i][1:200], 'r')
            lineList.append(line)
            axes[i].set_title(typeList[i] + ' max : ' + str(yfList[i][0]), fontsize=12, color='k')
            axes[i].set_ylim(0, 10)
        plt.xlabel('频率（HZ）')
        plt.legend(tuple(lineList), tuple(typeList), loc='lower right')
        plt.show()
            # plt.figure(figsize=(10,4))
            # plt.plot(np.array([i for i in range(len(wpsList))]), wpsList, 'r')
            # minH = min(wpsList[1000], wpsList[len(wpsList) - 1000])
            # maxH = max(wpsList[1000], wpsList[len(wpsList) - 1000])
            # plt.vlines(x=[1000, len(wpsList) - 1000], ymin=minH - 0.5 * minH, ymax=maxH + 0.5 * maxH,color='k')
            # plt.show()
            # normalization_half_y = np.add(np.array(yf1)[0 : min(len(yf1), len(normalization_half_y))], normalization_half_y[0 : min(len(yf1), len(normalization_half_y))])
        # plt.figure(figsize=(10,4))
        #
        # plt.plot(np.array([i + 1 for i in range(199)]), normalization_half_y[1:200] / len(pointList), 'blue')
        # plt.title(type[typeIndex] + ' : 单边振幅谱(归一化)', fontsize=9, color='blue')
        # typeIndex += 1
        # plt.show()
