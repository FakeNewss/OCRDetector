from WpsCal import *
from scipy import signal

def getWpsData(wpsList, wpsFilePath, cnt):
    file = open(wpsFilePath, 'r')
    dataList = []
    while True:
        line = file.readline()
        if line == None or cnt < 0:
            break
        print(line)
        cnt -= 1
        data =line[1:-2].split(', ')
        dataList.append(data)

    for data in dataList:
        data[4:] = map(eval, data[4:])
        data[0:4] = map(eval, data[0:4])
        wps = np.array(data)
        # wps = waveletSmooth(wps[3:], 0.1)
        wpsList.append(wps)
        # fig = plt.figure(figsize = (10,4), dpi=300)
        # plt.plot(wpsList[3:2000], 'r')
        # plt.show()
    return wpsList
def getBedtoolsWpsData(wpsFilePath, cnt):
    file = open(wpsFilePath, 'r')
    dataList = []
    while True:
        line = file.readline()
        if line == None or cnt < 0:
            break
        cnt -= 1
        data = line[:-1].split('\t')
        dataList.append(data)
    dataList = np.array(dataList, dtype=int)
    return dataList
class kalman_filter:
    def __init__(self, Q, R):
        self.Q = Q
        self.R = R

        self.P_k_k1 = 1
        self.Kg = 0
        self.P_k1_k1 = 1
        self.x_k_k1 = 0
        self.ADC_OLD_Value = 0
        self.Z_k = 0
        self.kalman_adc_old = 0

    def kalman(self, ADC_Value):
        self.Z_k = ADC_Value

        # if (abs(self.kalman_adc_old-ADC_Value)>=60):
        # self.x_k1_k1= ADC_Value*0.382 + self.kalman_adc_old*0.618
        # else:
        self.x_k1_k1 = self.kalman_adc_old;

        self.x_k_k1 = self.x_k1_k1
        self.P_k_k1 = self.P_k1_k1 + self.Q

        self.Kg = self.P_k_k1 / (self.P_k_k1 + self.R)

        kalman_adc = self.x_k_k1 + self.Kg * (self.Z_k - self.kalman_adc_old)
        self.P_k1_k1 = (1 - self.Kg) * self.P_k_k1
        self.P_k_k1 = self.P_k1_k1

        self.kalman_adc_old = kalman_adc

        return kalman_adc

def average_fft(x, fft_size):
    n = len(x) // fft_size * fft_size
    tmp = x[:n].reshape(-1, fft_size)
    tmp *= signal.hann(fft_size, sym=0)
    xf = np.abs(np.fft.rfft(tmp)/fft_size)
    avgf = np.average(xf, axis=0)
    return 20*np.log10(avgf)

def fft_combine(freqs, n, loops=1):
    length = len(freqs) * loops
    data = np.zeros(length)
    index = loops * np.arange(0, length, 1.0) / length * (2 * np.pi)
    for k, p in enumerate(freqs[:n]):
        if k != 0: p *= 2 # 除去直流成分之外，其余的系数都*2
        data += np.real(p) * np.cos(k*index) # 余弦成分的系数为实数部
        data -= np.imag(p) * np.sin(k*index) # 正弦成分的系数为负的虚数部
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
    yf = abs(np.fft.rfft(yt)/fft_size)
    freqs = np.linspace(0, 1.0*sampling_rate/2, 1.0*fft_size/2+1)
    return freqs, yf



if __name__ == '__main__':
    # wpsFilePath = '/mnt/X500/farmers/chenhx/02.project/hexiaoti/05.NDR_chr12/cellNornal.bam.list.result.wps.xls'
    wpsFilePath = '/home/chenlb/WPSCalProject/ndr.txt'
    wpsList = []
    dataList = getBedtoolsWpsData('/mnt/X500/farmers/chenlb/CellData/IH01/1.txt', 100000)
    plt.plot(dataList[35000:36000,1], dataList[35000:36000,2])
    plt.show()

    # wpsList = getWpsData(wpsList, wpsFilePath, 10)
    # kalman_filter = kalman_filter(0.01, 1)
    # for wps in wpsList:
    #     peaksList = []
    #     start = wps[0]
    #     end = wps[1]
    #     step = 4000
    #     # start = int(wps[1])
    #     # x = np.linspace(0)
    #     rawData = np.array(wps[2:])
    #     wpsData = np.array(wps[2:])
    #     fiterSize = len(wpsData)
    #     nyq = fiterSize * 0.5
    #     x = np.linspace(0, len(wpsData) - 4, fiterSize)
    #     freqs1, yf1 = fft(wpsData, fiterSize)
    #     data1 = np.fft.ifft(yf1)
    #     # b, a = signal.iirdesign(1000 / 4000.0, 1100 / 4000.0, 1, 10, 0, "cheby1")
    #     b, a = signal.butter(4, 110 / nyq, "lowpass")  # lowpass
    #     # 阶数；最大纹波允许低于通频带中的单位增益。以分贝表示，以正数表示；频率(Hz)/奈奎斯特频率（采样率*0.5）
    #     b, a = signal.cheby1(4, 5, 110 / nyq, "lowpass")  # lowpass
    #
    #     wpsData = signal.filtfilt(b, a, wpsData)
    #     # xf = average_fft(y, 20000)
    #     freqs2, yf2 = fft(wpsData, fiterSize)
    #
    #     fig, axes = plt.subplots(2, 1, figsize=(10,6))
    #     axes[0].plot(freqs1[0:400], yf1[0:400])
    #     axes[1].plot(freqs2[0:400], yf2[0:400])
    #     plt.show()
    #     step2 = 2500
    #     startIndex = start - start % 20000
    #     for i in range(0, len(wpsData), step2):
    #         fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    #         axes[0].plot(x[i:i + step2] + startIndex, wpsData[i:i + step2], 'r')
    #         axes[0].vlines(np.array([start, end]), ymin=np.array(wpsData[np.array([start % 20000, end % 20000])]) - 30,
    #                        ymax=np.array(wpsData[np.array([start % 20000, end % 20000])]) + 30, color='k')
    #         axes[1].plot(x[i:i + step2] + startIndex, rawData[i:i + step2], 'b')
    #         axes[1].vlines(np.array([start, end]), ymin=np.array(wpsData[np.array([start % 20000, end % 20000])]) - 30,
    #                        ymax=np.array(wpsData[np.array([start % 20000, end % 20000])]) + 30, color='k')
    #         plt.show()

        # for i in range(0, 25, 2):
        #     index, data = fft_combine(xf, i + 1, 2)  # 计算两个周期的合成波形
        #     axes[1].plot(data, label="N=%s" % i)
        # adc_filter_1 = []
        # for i in range(len(wpsData)):
        #     adc_filter_1.append(kalman_filter.kalman(wpsData[i]))
        #
        # fig, axes = plt.subplots(2, 1, figsize=(10, 5))
        # axes[0].plot(np.array([i for i in range(2000)]), wpsData[0:2000], 'r')
        # axes[1].plot(np.array([i for i in range(2000)]), adc_filter_1[0:2000], 'k')
        # plt.show()
        # wpsFFTData = np.fft.rfft(wpsData)
        # print(wpsFFTData[0:100])

        # wpsData = np.array(adc_filter_1)

        # peaks = scipy_signal_find_peaks(wpsData, height=0.05, distance=40, prominence=0.1, width=[30, 170])
        # peakObjectList = getValley(wpsData, wpsData, peaks[1], 4)
        # mergeValley(peakObjectList)
        # peaks.append(peakObjectList)
        # peaksList.append(peaks)
        # peaksList.append(peaks)

        # fft_size = 512
        #
        # wpsfft = np.fft.rfft(wpsData)/fft_size



        # ndrInformation_Smooth, ndrObjectList = findNDR(x, start, wps[0], peakObjectList, wpsData, wpsData,
        #                                                label='平滑算法数据', color='b',
        #                                                smoothMethod='平滑算法', peakDisThreshold=230)
        # drawPeaks(peaksList, 120, x, start)

        # for index in range(3, 20003, step):
        #     fig = plt.figure(figsize=(10,5))
        #     plt.plot([int(wps[1]) + i for i in range(index, index + step)], wps[index : index + step],color='r')
        #     title('chr : ' + str(wps[0]) + ' start : ' + str(wps[1] + index) + ' end :' + str(wps[2] + index + step))
        #     plt.show()
