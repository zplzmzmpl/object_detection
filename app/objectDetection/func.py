import cv2
import numpy as np
import torch

np.random.seed(0)


def rgb2gray(img):
    # Grayscale
    gray = 0.2126 * img[..., 2] + 0.7152 * img[..., 1] + 0.0722 * img[..., 0]
    return gray

'''
用来计算两个矩形的交并比（Intersection over Union，IoU）的。交并比是目标检测和图像分割中常用的评估指标。
函数iou接收两个矩形的坐标（左上角和右下角）作为输入，并返回它们的交并比。

    首先，计算两个矩形的面积。
    然后，计算两个矩形的交集区域的左上角和右下角的坐标。
    根据交集区域的坐标，计算交集区域的宽度和高度，并计算交集区域的面积。
    最后，计算交并比，即交集区域的面积除以两个矩形面积的和减去交集区域的面积。

返回计算得到的交并比。
'''

def iou(a, b):
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    iou_x1 = np.maximum(a[0], b[0])
    iou_y1 = np.maximum(a[1], b[1])
    iou_x2 = np.minimum(a[2], b[2])
    iou_y2 = np.minimum(a[3], b[3])
    iou_w = max(iou_x2 - iou_x1, 0)
    iou_h = max(iou_y2 - iou_y1, 0)
    area_iou = iou_w * iou_h
    iou = area_iou / (area_a + area_b - area_iou)
    return iou

'''
方向梯度直方图（Histogram of Oriented Gradients，HOG）特征提取算法。HOG是一种在计算机视觉和图像处理中用于对象检测的特征描述子。

    首先，计算输入图像的梯度和梯度幅度。梯度是通过对图像在x和y方向上的差分来计算的。梯度幅度是梯度的模。
    然后，计算梯度的方向。这里使用的是arctan函数，它返回一个数的反正切值。
    接着，将梯度的方向映射到0到8的范围，这8个值代表了0到180度的9个方向。
    然后，计算每个8x8的区域的梯度直方图。直方图的每个bin代表一个方向（0到8），其值是该方向上的梯度幅度的和。
    最后，对每个8x8的区域的直方图进行归一化处理。归一化是为了使不同大小的图像或不同位置的图像具有可比的特征。

函数返回得到的HOG特征。
'''
def hog(gray):
    h, w = gray.shape
    # Magnitude and gradient
    gray = np.pad(gray, (1, 1), 'edge')

    gx = gray[1:h+1, 2:] - gray[1:h+1, :w]
    gy = gray[2:, 1:w+1] - gray[:h, 1:w+1]
    gx[gx == 0] = 0.000001

    mag = np.sqrt(gx ** 2 + gy ** 2)
    gra = np.arctan(gy / gx)
    gra[gra<0] = np.pi / 2 + gra[gra < 0] + np.pi / 2

    # Gradient histogram
    gra_n = np.zeros_like(gra, dtype='uint8')

    d = np.pi / 9
    for i in range(9):
        gra_n[np.where((gra >= d * i) & (gra <= d * (i+1)))] = i

    N = 8
    HH = h // N
    HW = w // N
    Hist = np.zeros((HH, HW, 9), dtype=np.float32)
    for y in range(HH):
        for x in range(HW):
            for j in range(N):
                for i in range(N):
                    Hist[y, x, gra_n[y*4+j, x*4+i]] += mag[y*4+j, x*4+i]

    ## Normalization
    C = 3
    eps = 1
    for y in range(HH):
        for x in range(HW):
            #for i in range(9):
            Hist[y, x] /= np.sqrt(np.sum(Hist[max(y-1,0):min(y+2, HH), max(x-1,0):min(x+2, HW)] ** 2) + eps)

    return Hist

def resize(img, h, w):
    _h, _w  = img.shape
    ah = 1. * h / _h
    aw = 1. * w / _w
    y = np.arange(h).repeat(w).reshape(w, -1)
    x = np.tile(np.arange(w), (h, 1))
    y = (y / ah)
    x = (x / aw)

    ix = np.floor(x).astype(np.int32)
    iy = np.floor(y).astype(np.int32)
    ix = np.minimum(ix, _w-2)
    iy = np.minimum(iy, _h-2)

    dx = x - ix
    dy = y - iy

    out = (1-dx) * (1-dy) * img[iy, ix] + dx * (1 - dy) * img[iy, ix+1] + (1 - dx) * dy * img[iy+1, ix] + dx * dy * img[iy+1, ix+1]
    out[out>255] = 255

    return out


class NN:
    def __init__(self, ind=2, w=64, w2=64, outd=1, lr=0.1):
        self.w1 = np.random.normal(0, 1, [ind, w])
        self.b1 = np.random.normal(0, 1, [w])
        self.w2 = np.random.normal(0, 1, [w, w2])
        self.b2 = np.random.normal(0, 1, [w2])
        self.wout = np.random.normal(0, 1, [w2, outd])
        self.bout = np.random.normal(0, 1, [outd])
        self.lr = lr

    def forward(self, x):
        self.z1 = x
        self.z2 = sigmoid(np.dot(self.z1, self.w1) + self.b1)
        self.z3 = sigmoid(np.dot(self.z2, self.w2) + self.b2)
        self.out = sigmoid(np.dot(self.z3, self.wout) + self.bout)
        return self.out

    def train(self, x, t):
        # backpropagation output layer
        #En = t * np.log(self.out) + (1-t) * np.log(1-self.out)
        En = (self.out - t) * self.out * (1 - self.out)
        grad_wout = np.dot(self.z3.T, En)
        grad_bout = np.dot(np.ones([En.shape[0]]), En)
        self.wout -= self.lr * grad_wout
        self.bout -= self.lr * grad_bout

        # backpropagation inter layer
        grad_u2 = np.dot(En, self.wout.T) * self.z3 * (1 - self.z3)
        grad_w2 = np.dot(self.z2.T, grad_u2)
        grad_b2 = np.dot(np.ones([grad_u2.shape[0]]), grad_u2)
        self.w2 -= self.lr * grad_w2
        self.b2 -= self.lr * grad_b2

        grad_u1 = np.dot(grad_u2, self.w2.T) * self.z2 * (1 - self.z2)
        grad_w1 = np.dot(self.z1.T, grad_u1)
        grad_b1 = np.dot(np.ones([grad_u1.shape[0]]), grad_u1)
        self.w1 -= self.lr * grad_w1
        self.b1 -= self.lr * grad_b1

def sigmoid(x):
    return 1. / (1. + np.exp(-x))


'''
创建一个空的数据库（db），其大小为Crop_num x (F_n+1)。
1，对于每个要裁剪的区域，随机选择一个左上角的点（x1, y1），然后计算出右下角的点（x2, y2）。
2，计算裁剪区域和ground truth（gt）的交并比（IoU）。如果IoU大于等于0.5，那么就将该区域标记为正样本（label=1），否则标记为负样本（label=0）。
3，将裁剪区域的图像用红色或蓝色矩形框起来，表示其是正样本还是负样本。
4，将裁剪区域的图像缩放到指定的大小（H_size x H_size），然后计算出该区域的HOG特征。
5，将HOG特征和标签存储到数据库中。
'''

# crop and create database
def crop_db(Crop_num, img, L, H_size, F_n, gt):
    gray = rgb2gray(img)
    H, W = gray.shape
    db = np.zeros((Crop_num, F_n+1))
    for i in range(Crop_num):
        x1 = np.random.randint(W-L)
        y1 = np.random.randint(H-L)
        x2 = x1 + L
        y2 = y1 + L
        crop = np.array((x1, y1, x2, y2))

        _iou = iou(gt, crop)

        if _iou >= 0.5:
            cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 1)
            label = 1
        else:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,255), 1)
            label = 0

        crop_area = gray[y1:y2, x1:x2]
        crop_area = resize(crop_area, H_size, H_size)
        _hog = hog(crop_area)

        db[i, :F_n] = _hog.ravel()
        db[i, -1] = label
    cv2.imwrite("segment.jpg", img)
    return db



# sliding window
def slide_window(img, nn, detects, recs, H_size, step, flag, threshold):
    gray = rgb2gray(img)
    H,W = gray.shape
    for y in range(0, H, step):
        for x in range(0, W, step):
            for rec in recs:
                dh = int(rec[0] // 2)
                dw = int(rec[1] // 2)
                x1 = max(x-dw, 0)
                x2 = min(x+dw, W)
                y1 = max(y-dh, 0)
                y2 = min(y+dh, H)
                region = gray[max(y-dh,0):min(y+dh,H), max(x-dw,0):min(x+dw,W)]
                region = resize(region, H_size, H_size)
                region_hog = hog(region).ravel()

                if flag:
                    print('use gpu...')
                    score = nn(torch.tensor(region_hog, dtype=torch.float32))
                    score = torch.sigmoid(score)
                else:
                    score = nn.forwad(region_hog)
                # print(score)
                if score >= threshold:
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 1)
                    detects = np.vstack((detects, np.array((x1, y1, x2, y2,score.item()))))
    cv2.imwrite("slid_win_result.jpg",img)
    return detects


'''
1，定义了一个函数nms，该函数接受以下参数：
    _bboxes：一个二维数组，每一行代表一个检测框，格式为[leftTopX, leftTopY, w, h, score]。
    iou_th：IoU（Intersection over Union）阈值，用于判断两个检测框是否重叠。
    select_num：选择的检测框数量，如果为None，则不限制数量。
    prob_th：置信度阈值，只有大于等于该阈值的检测框才会被选择。如果为None，则不限制置信度。
2，创建一个副本bboxes，并计算每个检测框的宽度和高度。
3，根据检测框的得分（score）对bboxes进行降序排序，并获取排序后的索引。
4，初始化一个空列表return_inds，用于存储最终选择的检测框的索引。
5，创建一个副本unselected_inds，用于存储未被选择的检测框的索引。
6，进入一个循环，直到unselected_inds为空。在循环中，首先找到得分最高的检测框，将其索引添加到return_inds，并从unselected_inds中删除。
7，然后，计算该检测框与其他检测框的IoU，并删除IoU大于等于iou_th的检测框。
如果prob_th不为None，则根据置信度阈值过滤return_inds。
如果select_num不为None，则根据select_num限制return_inds的长度。

'''
# Non-maximum suppression
def nms(_bboxes, iou_th=0.5, select_num=None, prob_th=None):
    bboxes = _bboxes.copy()

    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]

    # Sort by bbox's score. High -> Low
    sort_inds = np.argsort(bboxes[:, -1])[::-1]

    processed_bbox_ind = []
    return_inds = []

    unselected_inds = sort_inds.copy()

    while len(unselected_inds) > 0:
        process_bboxes = bboxes[unselected_inds]
        argmax_score_ind = np.argmax(process_bboxes[::, -1])
        max_score_ind = unselected_inds[argmax_score_ind]
        return_inds += [max_score_ind]
        unselected_inds = np.delete(unselected_inds, argmax_score_ind)

        base_bbox = bboxes[max_score_ind]
        compare_bboxes = bboxes[unselected_inds]

        base_x1 = base_bbox[0]
        base_y1 = base_bbox[1]
        base_x2 = base_bbox[2] + base_x1
        base_y2 = base_bbox[3] + base_y1
        base_w = np.maximum(base_bbox[2], 0)
        base_h = np.maximum(base_bbox[3], 0)
        base_area = base_w * base_h

        # compute iou-area between base bbox and other bboxes
        iou_x1 = np.maximum(base_x1, compare_bboxes[:, 0])
        iou_y1 = np.maximum(base_y1, compare_bboxes[:, 1])
        iou_x2 = np.minimum(base_x2, compare_bboxes[:, 2] + compare_bboxes[:, 0])
        iou_y2 = np.minimum(base_y2, compare_bboxes[:, 3] + compare_bboxes[:, 1])
        iou_w = np.maximum(iou_x2 - iou_x1, 0)
        iou_h = np.maximum(iou_y2 - iou_y1, 0)
        iou_area = iou_w * iou_h

        compare_w = np.maximum(compare_bboxes[:, 2], 0)
        compare_h = np.maximum(compare_bboxes[:, 3], 0)
        compare_area = compare_w * compare_h

        # bbox's index which iou ratio over threshold is excluded
        all_area = compare_area + base_area - iou_area
        iou_ratio = np.zeros((len(unselected_inds)))
        iou_ratio[all_area < 0.9] = 0.
        _ind = all_area >= 0.9
        iou_ratio[_ind] = iou_area[_ind] / all_area[_ind]

        unselected_inds = np.delete(unselected_inds, np.where(iou_ratio >= iou_th)[0])

    if prob_th is not None:
        preds = bboxes[return_inds][:, -1]
        return_inds = np.array(return_inds)[np.where(preds >= prob_th)[0]].tolist()

    # pick bbox's index by defined number with higher score
    if select_num is not None:
        return_inds = return_inds[:select_num]

    return return_inds


# Evaluation
# Recall, Precision, F-score
def evaluation(GT, detects,iou_th):
    Rs = np.zeros((len(GT)))
    Ps = np.zeros((len(detects)))
    for i, g in enumerate(GT):
        iou_x1 = np.maximum(g[0], detects[:, 0])
        iou_y1 = np.maximum(g[1], detects[:, 1])
        iou_x2 = np.minimum(g[2], detects[:, 2])
        iou_y2 = np.minimum(g[3], detects[:, 3])
        iou_w = np.maximum(0, iou_x2 - iou_x1)
        iou_h = np.maximum(0, iou_y2 - iou_y1)
        iou_area = iou_w * iou_h
        g_area = (g[2] - g[0]) * (g[3] - g[1])
        d_area = (detects[:, 2] - detects[:, 0]) * (detects[:, 3] - detects[:, 1])
        ious = iou_area / (g_area + d_area - iou_area)

        Rs[i] = 1 if len(np.where(ious >= iou_th)[0]) > 0 else 0
        Ps[ious >= iou_th] = 1

    R = np.sum(Rs) / len(Rs)
    P = np.sum(Ps) / len(Ps)
    F = (2 * P * R) / (P + R)

    print("Recall >> {:.2f} ({} / {})".format(R, np.sum(Rs), len(Rs)))
    print("Precision >> {:.2f} ({} / {})".format(P, np.sum(Ps), len(Ps)))
    print("F-score >> ", F)

    # Open a file in write mode
    f = open("output.txt", "w")

    # Write the output to the file
    f.write("Recall >> {:.2f} ({} / {})\n".format(R, np.sum(Rs), len(Rs)))
    f.write("Precision >> {:.2f} ({} / {})\n".format(P, np.sum(Ps), len(Ps)))
    f.write("F-score >> {}\n".format(F))

    # Don't forget to close the file
    f.close()

    return Ps




## mAP
def cal_mAP(detects, Ps):
    mAP = 0.
    for i in range(len(detects)):
        mAP += np.sum(Ps[:i]) / (i + 1) * Ps[i]
    mAP /= np.sum(Ps)

    print("mAP >>", mAP)

    f = open("output.txt", "a")
    f.write("mAP >> {}".format(mAP))
    f.close()



# Display
def dsiplay_final(img, detects, GT, Ps):
    for i in range(len(detects)):
        v = list(map(int, detects[i, :4]))
        if Ps[i] > 0:
            cv2.rectangle(img, (v[0], v[1]), (v[2], v[3]), (0,0,255), 1)
        else:
            cv2.rectangle(img, (v[0], v[1]), (v[2], v[3]), (255,0,0), 1)
        cv2.putText(img, "{:.2f}".format(detects[i, -1]), (v[0], v[1]+9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,0,255), 1)

    for g in GT:
        cv2.rectangle(img, (g[0],g[1]), (g[2], g[3]), (0,0,0), 1)

    cv2.imwrite("final.jpg",img)
