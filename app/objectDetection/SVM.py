import numpy as np
import os
import random
from HOG import Hog_descriptor
from PIL import Image
import cv2

def selectJrand(i, m):
    j = i  # 随机找一个不等于i的j
    while (j == i):
        j = int(np.random.uniform(0, m))
    return j


def kernelTrans(X, A, kTup):  # 计算核函数值，将样本映射到高维空间
    m, n = np.shape(X)
    K = np.mat(np.zeros((m, 1)))
    if kTup[0] == 'lin':
        K = X * A.T  # linear kernel 线性核
    elif kTup[0] == 'rbf':  # 高斯核函数
        for j in range(m):
            deltaRow = X[j, :] - A
            K[j] = deltaRow * deltaRow.T
        K = np.exp(K / (-1 * kTup[1] ** 2))  # divide in NumPy is element-wise not matrix like Matlab
    else:
        raise NameError('Houston We Have a Problem -- That Kernel is not recognized')
    return K


class optStruct:
    def __init__(self, dataMatIn, classLabels, C, toler, kTup):  # Initialize the structure with the parameters
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = np.shape(dataMatIn)[0]
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0
        self.eCache = np.mat(np.zeros((self.m, 2)))  # first column is valid flag
        self.K = np.mat(np.zeros((self.m, self.m)))
        for i in range(self.m):  # 暴力n^2算出所有xTx对的核函数值
            self.K[:, i] = kernelTrans(self.X, self.X[i, :], kTup)


def calcEk(oS, k):
    fXk = float(np.multiply(oS.alphas, oS.labelMat).T * oS.K[:, k] + oS.b)
    Ek = fXk - float(oS.labelMat[k])
    return Ek


def selectJ(i, oS, Ei):  # this is the second choice -heurstic, and calcs Ej
    maxK = -1;
    maxDeltaE = 0;
    Ej = 0
    oS.eCache[i] = [1, Ei]  # set valid #choose the alpha that gives the maximum delta E
    validEcacheList = np.nonzero(oS.eCache[:, 0].A)[0]
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:  # loop through valid Ecache values and find the one that maximizes delta E
            if k == i: continue  # don't calc for i, waste of time
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE):
                maxK = k;
                maxDeltaE = deltaE;
                Ej = Ek
        return maxK, Ej
    else:  # in this case (first time around) we don't have any valid eCache values
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej


def updateEk(oS, k):  # after any alpha has changed update the new value in the cache
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]


def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


def innerL(i, oS):
    Ei = calcEk(oS, i)
    if ((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or (
            (oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):
        j, Ej = selectJ(i, oS, Ei)  # this has been changed from selectJrand
        alphaIold = oS.alphas[i].copy();
        alphaJold = oS.alphas[j].copy();
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H:
            # print("L==H")
            return 0
        eta = 2.0 * oS.K[i, j] - oS.K[i, i] - oS.K[j, j]  # changed for kernel
        if eta >= 0:
            # print("eta>=0")
            return 0
        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        updateEk(oS, j)  # added this for the Ecache
        if (abs(oS.alphas[j] - alphaJold) < 0.00001):
            # print("j not moving enough")
            return 0
        oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alphaJold - oS.alphas[j])  # update i by the same amount as j
        updateEk(oS, i)  # added this for the Ecache                    #the update is in the oppostie direction
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, i] - oS.labelMat[j] * (
                oS.alphas[j] - alphaJold) * oS.K[i, j]
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, j] - oS.labelMat[j] * (
                oS.alphas[j] - alphaJold) * oS.K[j, j]
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
            oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0


def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup=('lin', 0)):  # full Platt SMO
    oS = optStruct(np.mat(dataMatIn), np.mat(classLabels).transpose(), C, toler, kTup)
    iter = 0
    entireSet = True
    alphaPairsChanged = 0
    f = open("svm_output.txt", "w")

    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:  # go over all
            for i in range(oS.m):
                alphaPairsChanged += innerL(i, oS)
                # print("fullSet, iter: %d i:%d, pairs changed %d\n" % (iter, i, alphaPairsChanged))
                f.write("fullSet, iter: %d i:%d, pairs changed %d\n" % (iter, i, alphaPairsChanged))
            iter += 1
        else:  # go over non-bound (railed) alphas
            nonBoundIs = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, oS)
                # print("non-bound, iter: %d i:%d, pairs changed %d\n" % (iter, i, alphaPairsChanged))
                f.write("non-bound, iter: %d i:%d, pairs changed %d\n" % (iter, i, alphaPairsChanged))
            iter += 1
        if entireSet:
            entireSet = False  # toggle entire set loop
        elif (alphaPairsChanged == 0):
            entireSet = True
        # print("iteration number: %d\n" % iter)
        f.write("iteration number: %d\n" % iter)
    f.close()
    return oS.b, oS.alphas


def calcWs(alphas, dataArr, classLabels):
    X = np.mat(dataArr)
    labelMat = np.mat(classLabels).transpose()
    m, n = np.shape(X)
    w = np.zeros((1, n))  # 横向
    for i in range(m):
        w += np.multiply(alphas[i] * labelMat[i], X[i, :])
    return np.mat(w)


def SMO(dataSetIn, classLabels, C, toler, maxIter, kTup=('lin', 0)):
    # b, alphas = smoSimple(dataSetIn, classLabels, C, toler, maxIter)
    b, alphas = smoP(dataSetIn, classLabels, C, toler, maxIter, kTup)  # SVM 训练
    w = calcWs(alphas, dataSetIn, classLabels)  # 计算w
    return w, b, alphas  # 返回类型均为np.matrix   w(1,n),b(1,1),alpha(m,1)

def resize_img(infile, outfile):
    im = Image.open(infile)
    (x, y) = im.size  # read image size
    x_s = 64  # define standard width
    y_s = y * x_s // x  # calc height based on standard width
    out = im.resize((x_s, y_s), Image.ANTIALIAS)  # resize image with high-quality
    out.save(outfile)



#  读取训练集,args样本个数
def loadDataSet(pos_path,pos_count,neg_path,neg_count):
    dataSet = []
    label = []
    posImgs = os.listdir(pos_path)  # 正样本
    negImgs = os.listdir(neg_path)

    for i in range(min(len(posImgs),pos_count)):  # 读取正样本
        path = pos_path+ '/' + posImgs[i]
        img = cv2.imread(path, 0)  # 读取灰度图
        hog_vec, hog_img = Hog_descriptor(img).extract()  # 计算HOG特征描述子
        dataSet.append(hog_vec)
        label.append(1)

    for i in range(min(len(negImgs),neg_count)):  # 读取负样本
        path = neg_path+ '/' + negImgs[i]
        img = cv2.imread(path, 0)  # 读取灰度图
        hog_vec, hog_img = Hog_descriptor(img).extract()  # 计算HOG特征描述子
        dataSet.append(hog_vec)
        label.append(-1)
    return dataSet, label



def do_training(pos_path,pos_c,neg_path,neg_c):  # 训练产生w b alpha
    dataSet, label = loadDataSet(pos_path,pos_c,neg_path,neg_c)  # HOG特征向量的训练集

    ws, b, alpha = SMO(dataSetIn=dataSet, classLabels=label, C=0.6, toler=0.001, maxIter=60)  # SVM训练
    #  ws(1,n),b(1,1),alpha(m,1)    np.matrix
    fw = open('weights.txt', 'w')
    for i in range(ws.shape[1]):
        fw.write(str(ws[0, i]) + ' ')
    fw.write('\n' + str(b[0,0]) + '\n')
    for i in range(alpha.shape[0]):
        fw.write(str(alpha[i, 0]) + ' ')
    fw.write('\n')


# 给图片加框 https://www.jb51.net/article/155363.htm
def run(test_path,slide_len,width_step,hei):

    #  ws(1,n),b(1,1),alpha(m,1)    请保证ws,alpha为np.matrix
    ws = []
    fr = open('weights.txt')
    for i in fr.readline().strip().split(' '):
        ws.append(float(i))
    b=float(fr.readline().strip())
    fr.close()
    ws = np.mat(ws)

    #  下面进行分类

    test_img = cv2.imread(test_path)  # 原始彩色图
    test_img_gray = cv2.imread(test_path, 0)  # 灰度图

    height, width = test_img_gray.shape  # 高和宽
    # slide_len=4    # 滑动步长
    # width_step=100   # 窗口宽度步长
    win_sum=0; target = 0
    for i in range(0,height,slide_len):
        for j in range(0,width,slide_len):
            for wid in range(width_step,width,width_step):
                # hei = wid*ht
                if i+hei>=height or j+wid>=width: break
                win_sum+=1
                window = test_img_gray[i:i + hei, j:j + wid]  # 剪出窗口
                window = cv2.resize(window, (64, 128))  # 调整窗口像素为128*64

                hog_vec, hog_img = Hog_descriptor(window).extract()    # HOG
                fx = np.mat(hog_vec) * ws.T + b  # 预测
                if fx > 0:
                    target += 1
                    cv2.rectangle(test_img, (j, i), (j + wid, i + hei), (255, 0, random.randint(0,255)), 3)
                    # plot_compare_2img(window,hog_img)
    print('滑动步长%d,窗宽步长%d,检测窗口个数%d,检测到%d次目标' % (slide_len,width_step,win_sum,target))
    cv2.imwite("svm_result.jpg",test_img)
