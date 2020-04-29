import numpy as np
import scipy.signal
import matplotlib
from matplotlib import pyplot as plt
import pysam
from WpsCal import *
import peakutils
import numpy as np
import pylab as pl
from pykalman import KalmanFilter
from WpsCal import *
def getBamFileInformation2(contig, bamfile, start, end):
    '''
    得到bam文件的基础信息
    :param bamfile:bam文件操作符
    :return:start: bam中的reads最小的起点
            end: 文件中最远的fragment终点
    '''
    s = 100000000000;
    e = 0
    cnt = 0
    for data in bamfile.fetch(contig=contig, start=start, end=end):
        if cnt > 200:
            break
        cnt += 1
        if data.isize > 0 and data.isize < 1000:
            read = bamfile.mate(data)
            print(data.isize,' -- ',data.reference_start,' -- ',data.reference_end,' -- ',read.reference_end - data.reference_start,' -- ',data.mapping_quality)
    print('getBamFileInformation done')

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

def fft_combine(freqs, n, loops=1):
    length = len(freqs) * loops
    data = np.zeros(length)
    index = loops * np.arange(0, length, 1.0) / length * (2 * np.pi)
    for k, p in enumerate(freqs[:n]):
        if k != 0: p *= 2  # 除去直流成分之外，其余的系数都*2
        data += np.real(p) * np.cos(k * index)  # 余弦成分的系数为实数部
        data -= np.imag(p) * np.sin(k * index)  # 正弦成分的系数为负的虚数部
    return index, data

    # 产生size点取样的三角波，其周期为1
def triangle_wave(size):
    x = np.arange(0, 1, 1.0 / size)
    y = np.where(x < 0.5, x, 0)
    y = np.where(x >= 0.5, 1 - x, y)
    return x, y



if __name__ == '__main__':
    # list = [164, 172, 116, 213, 329, 168, 329, 115, 114, 149, 200, 163, 184, 183, 223, 172, 245, 160, 203, 202, 195, 149, 156, 159, 211, 155, 187, 396, 134, 255, 196, 132, 107, 99, 104, 117, 180, 182, 207, 224, 137, 116, 176, 116, 108, 77, 114, 114, 159, 84, 95, 95, 112, 152, 180, 116, 117, 158, 156, 61, 144, 376, 200, 113, 259, 142, 166, 275, 183, 168, 120, 405, 117, 109, 94, 141, 392, 136, 74, 148, 169, 164, 180, 328, 228, 196, 144, 225, 199, 197, 185, 167, 192, 192, 164, 190, 235, 142, 132, 205, 209, 111, 95, 177, 114, 96, 140, 212, 173, 242, 168, 228, 115, 110, 132, 151, 144, 194, 137, 189, 176, 172, 166, 166, 171, 243, 210, 121, 86, 170, 170, 141, 174, 121, 157, 166, 88, 280, 200, 109, 160, 163, 112, 115, 156, 220, 133, 180, 97, 145, 193, 184, 141, 137, 136, 185, 196, 156, 446, 177, 125, 118, 217, 193, 180, 236, 172, 172, 177, 164, 248, 108, 109, 196, 168, 186, 187, 153, 92, 143, 0, 144, 272, 102, 98, 160, 133, 90, 88, 174, 304, 88, 136, 113, 100, 132, 133, 151, 182, 92, 104, 172, 104, 102, 100, 144, 168, 136, 191, 216, 155, 214, 203, 103, 100, 153, 188, 192, 188, 169, 230, 84, 85, 109, 128, 143, 76, 148, 175, 101, 97, 316, 213, 93, 325, 137, 76, 147, 216, 184, 159, 218, 143, 175, 196, 168, 201, 182, 193, 194, 141, 165, 164, 78, 171, 151, 147, 88, 178, 167, 167, 166, 116, 91, 100, 124, 272, 125, 93, 176, 148, 332, 191, 192, 151, 124, 97, 188, 197, 196, 254, 143, 94, 82, 279, 168, 151, 140, 252, 120, 204, 156, 288, 214, 218, 157, 176, 175, 187, 156, 176, 164, 152, 256, 190, 179, 172, 172, 201, 200, 156, 150, 242, 165, 207, 179, 136, 152, 136, 173, 196, 120, 109, 108, 159, 123, 88, 96, 123, 176, 98, 103, 101, 98, 97, 183, 204, 144, 168, 154, 154, 196, 115, 188, 135, 131, 264, 145, 141, 168, 120, 113, 156, 296, 160, 99, 98, 168, 176, 104, 116, 292, 157, 96, 96, 299, 139, 161, 233, 136, 116, 199, 164, 161, 232, 140, 205, 174, 125, 117, 97, 93, 88, 224, 221, 98, 111, 155, 158, 143, 75, 150, 259, 151, 156, 224, 156, 165, 177, 111, 112, 128, 224, 111, 216, 1, 214, 172, 223, 175, 147, 261, 162, 191, 156, 121, 188, 239, 139, 93, 153, 219, 178, 112, 108, 124, 213, 184, 204, 116, 101, 269, 101, 90, 169, 196, 197, 249, 140, 92, 108, 144, 127, 68, 189, 187, 171, 164, 127, 104, 138, 199, 135, 276, 115, 168, 173, 186, 151, 194, 203, 279, 104, 101, 177, 212, 189, 108, 163, 104, 160, 159, 199, 175, 86, 92, 104, 132, 132, 101, 107, 163, 252, 172, 124, 196, 112, 212, 176, 173, 207, 140, 202, 112, 100, 140, 123, 116, 268, 123, 175, 137, 134, 93, 122, 102, 341, 179, 167, 168, 193, 256, 106, 293, 220, 167, 212, 170, 174, 131, 175, 334, 215, 144, 233, 125, 208, 211, 191, 233, 229, 171, 187, 69, 138, 104, 91, 83, 122, 123, 126, 122, 93, 105, 171, 170, 137, 137, 133, 176, 185, 190, 206, 160, 276, 145, 202, 198, 140, 180, 193, 188, 220, 180, 180, 184, 200, 180, 96, 145, 160, 132, 102, 144, 166, 123, 223, 181, 180, 80, 125, 182, 182, 181, 208, 167, 184, 227, 187, 212, 179, 117, 161, 2, 163, 144, 87, 140, 188, 184, 180, 195, 223, 177, 146, 144, 208, 200, 171, 100, 145, 184, 80, 120, 118, 141, 144, 106, 111, 124, 188, 98, 125, 228, 176, 215, 179, 211, 168, 236, 165, 169, 170, 124, 195, 107, 107, 95, 96, 101, 144, 145, 141, 238, 120, 281, 98, 94, 199, 152, 176, 47]
    # list = np.array(list*10)
    # KernelDensityEstimate(kernel = 'gaussian',bandwidth = 20, dataList = list, start = 50, end = 350, point = 10000, bin = 10, maxYlim = 0.05)
    # 取FFT计算的结果freqs中的前n项进行合成，返回合成结果，计算loops个周期的波形
    fft_size = 256

    # 计算三角波和其FFT
    x, y = triangle_wave(fft_size)
    sampling_rate = len(y)
    fy = np.fft.rfft(y) / fft_size
    freqs = np.linspace(0, sampling_rate / 2, fft_size / 2 + 1)
    xfp = 20 * np.log10(np.clip(np.abs(fy), 1e-20, 1e100))

    # 绘制三角波的FFT的前20项的振幅，由于不含下标为偶数的值均为0， 因此取
    # log之后无穷小，无法绘图，用np.clip函数设置数组值的上下限，保证绘图正确
    fig,axes = plt.subplots(2, 1)
    axes[0].plot(x[:fft_size], y)
    axes[1].plot(freqs, xfp)

    plt.xlabel("frequency bin")
    plt.ylabel("power(dB)")
    plt.title("FFT result of triangle wave")

    # 绘制原始的三角波和用正弦波逐级合成的结果，使用取样点为x轴坐标
    plt.figure()
    plt.plot(y, label="original triangle", linewidth=2)
    for i in [0, 1, 3, 5, 7, 9]:
        index, data = fft_combine(fy, i + 1, 2)  # 计算两个周期的合成波形
        plt.plot(data, label="N=%s" % i)
    plt.legend()
    plt.title("partial Fourier series of triangle wave")
    plt.show()