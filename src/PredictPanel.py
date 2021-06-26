# import keras
import os
import cv2
import scipy
import numpy as np
# import keras
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from cleanlab.classification import LearningWithNoisyLabels
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from keras import *
# from keras import *
import pandas as pd
import matplotlib
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn import svm
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.base import BaseEstimator
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier,ExtraTreeClassifier
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score, cross_validate
import warnings
import pickle
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from functools import reduce
# warnings.filterwarnings('ignore')
import seaborn as sns
from sklearn import manifold
from sklearn.model_selection import GridSearchCV
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
matplotlib.rcParams['font.family']='STSong'#修改了全局变量
import random
import copy
from functools import reduce
warnings.filterwarnings('ignore')
from sklearn.metrics import classification_report

def transferYPred(y_pred):
    print(y_pred)
    for i in range(len(y_pred)):
        max_value=max(y_pred[i])
        for j in range(len(y_pred[i])):
            if max_value==y_pred[i][j]:
                y_pred[i][j]=1
            else:
                y_pred[i][j]=0
    print(y_pred)
    return y_pred

def getResult(testX, testY, clf, title):
    y_pred = clf.predict(testX)
    y_prob = clf.predict_proba(testX)
    # print("precision = ", precision_score(testY, y_pred))
    # print("recall = ", recall_score(testY, y_pred))
    # print("f1-score = ", f1_score(testY, y_pred))
    plt.title(title + ':特征提取+逻辑回归结果')
    accscore = str(accuracy_score(testY, y_pred))[0:6]
    aucscore = str(roc_auc_score(testY, y_prob, multi_class='ovr'))[0:6]
    recallsocre = str(recall_score(testY, y_pred, average='weighted'))[0:6]
    f1score = str(f1_score(testY, y_pred, average = 'weighted'))[0:6]
    print("auc = ", aucscore)
    print("acc = ", accscore)
    print("recall = ", recallsocre)
    print("f1-score = ", f1score)

    # print('-----------------特征提取+逻辑回结果-----------------')
    report = classification_report(testY, y_pred)
    print(report)
    conf_matrix = confusion_matrix(testY, y_pred)
    # print("confusion matrix = \n", conf_matrix)
    # sns.heatmap(conf_matrix, fmt='.20g', square=True, annot=True)
    # plt.title('特征提取+逻辑回归结果:测试集ACC = '+ str(accuracy_score(testY, y_pred))[0:5] + ' ; 测试集AUC=' + str(roc_auc_score(testY, y_prob, multi_class='ovr'))[0:5])

    # plt.title(title+':特征提取+逻辑回归结果:测试集ACC = ' + accscore)
    # plt.show()
    # plt.close()
    return [float(accscore), float(aucscore) , float(recallsocre), float(f1score)]
    # with open('rf.pickle', 'wb') as fw:
    #     pickle.dump(clf, fw)
    # pickle.dump(std, open('scaler.pkl','wb'))


def drawRes(x, y, xNoise, yNoise, title):
    plt.plot(x, y)
    plt.plot(xNoise, yNoise)
    plt.ylim(0, 1.0)
    plt.title(title)
    plt.legend('普通', '置信学习')
    plt.show()
    plt.close()

if __name__ == '__main__':
    data = pd.read_csv('./dataset/dataAll_range3.csv')
    data = data.dropna(axis=0, how='any')

    # data.columns = ['lines', 'label', 'depth', 'bigAreaDepth', 'smallDepth', 'height', 'width', 'angel', 'area',
    #                 'leftK', 'rightK', 'heightVar', 'widthVar', 'angelVar', 'areaVar', 'leftKVar', 'rightKVar',
    #                 'coefOneHot', 'leffCoef', 'rightCoef', 'allCoef', 'leftIntercept', 'rightIntercept', 'allIntercept',
    #                 'radianMid', 'radianAdd', 'intersectionX', 'intersectionY', 'vectorX', 'vectorY', 'chr', 'start', 'end']
    #, 'contig', 'start', 'end'
    # data = data[data['0'] != 2]
    # data = data[data['0'] != 3]

    # print(np.where(np.isnan(data)))
    Y = data.iloc[:, 1].values

    cnt = 0
    # for i in range(len(Y)):
    #     if Y[i] == 1 and cnt <640:
    #         Y[i] = 0
    #         cnt += 1

    # feat_labels = data.columns[3:6]
    X = np.hstack((data.iloc[:,4:6].values, data.iloc[:,-15:-3].values))


    # sns.distplot(X[:1])
    # plt.show()
    # print(X[0])
    # X =  pd.concat([data.iloc[:,19:20], data.iloc[:,22:23]], axis=1)
    # X =  pd.concat([data.iloc[:,2:5], data.iloc[:,17:-3]], axis=1)
    X = np.array(X, dtype=np.float)
    Y = np.array(Y, dtype=np.int)




    # trainX = std.transform(trainX)
    # testX = std.transform(testX)

    (trainX, testX, trainY, testY) = train_test_split(X, Y, test_size=0.3, random_state=150)

    std = StandardScaler()
    std = std.fit(trainX)
    trainX = std.transform(trainX)
    testX = std.transform(testX)
    # X = std.transform(X)

    #
    # f1_scoreList = []
    # drawT_SNE(data, X, Y, 3)
    # X_new = drawT_SNE(data, X, Y, n_com)
    # sns.heatmap(X)
    # plt.show()
    # plt.close()

    # X = std.transform(X)
    # one-hot编码
    # Y = keras.utils.to_categorical(Y)
    # 数据集切分

    # print(testY)
    # print(trainY)
    #    clf = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    # decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    # max_iter=-1, probability=False, random_state=None, shrinking=True,
    # tol=0.001, verbose=False)
    #    clf = RandomForestClassifier()
    #    clf.fit(trainX, trainY)

    # clf = LearningWithNoisyLabels(clf=    ())

    # kernel = C(1.0, (1e-3, 1e3)) * RBF(0, (0.5, 2))
    # clf = LearningWithNoisyLabels(clf=GaussianProcessClassifier(kernel= kernel, max_iter_predict=1000, multi_class='one_vs_rest'))

    # clf = GaussianProcessClassifier(kernel= kernel, multi_class='one_vs_rest')
    # clf = LearningWithNoisyLabels(GaussianProcessClassifier(kernel= kernel, multi_class='one_vs_rest'))
    # clf = LearningWithNoisyLabels(clf = RandomForestClassifier())
    # clf =  LogisticRegression(penalty="l1", solver="liblinear")
    # clf = GradientBoostingClassifier()
    # clf = SVC(probability=True)
    # clf = GaussianProcessClassifier(kernel=kernel, multi_class='one_vs_rest')

    ratioList = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    scoreList = []
    noiseScoreist = []
    for ratio in ratioList:
        clf = RandomForestClassifier()
        clfNoise = LearningWithNoisyLabels(clf=RandomForestClassifier())
        newTrainX = trainX
        newTrainY = copy.deepcopy(trainY)
        for i in range(len(newTrainX)):
            if (random.random() < ratio):
                while True:
                    noiseLabel = random.randint(1, 4) - 1;
                    # print('trainY[i] : ', newTrainY[i], 'noiseLabel :',noiseLabel)
                    if newTrainY[i] != noiseLabel:
                        newTrainY[i] = noiseLabel
                        break

        clf.fit(newTrainX, newTrainY)
        clfNoise.fit(newTrainX, newTrainY)
        # importances = clf.feature_importances_
        # indices = np.argsort(importances)[::-1]
        # for f in range(trainX.shape[1]):
        #     print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))

        scores= getResult(testX, testY, clf, '普通-' + str(ratio)) #accscore, aucscore, recallsocre, f1score
        noiseScores = getResult(testX,testY, clfNoise, '置信学习-' + str(ratio))
        scoreList.append(scores)
        noiseScoreist.append(noiseScores)
    scoreList = np.array(scoreList)
    noiseScoreist = np.array(noiseScoreist)
    titleList = ['accscore', 'aucscore', 'recallsocre', 'f1score']
    for i in range(len(scoreList[0])):
        drawRes(ratioList, scoreList[:,i], ratioList, noiseScoreist[:,i], str(ratioList[i]) + ':' + titleList[i])


