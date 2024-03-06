# This Python file uses the following encoding: utf-8
import sys

from PySide6.QtWidgets import QApplication, QWidget, QFileDialog
from PySide6.QtGui import QPixmap
# Important:
# You need to run the following command to generate the ui_form.py file
#     pyside6-uic form.ui -o ui_form.py, or
#     pyside2-uic form.ui -o ui_form.py

from ui_form import Ui_objDec
import cv2
import numpy as np
import func
import torch
from torch.autograd import Variable
from nn import CNN
import SVM
import HOG
import pre_trained

class objDec(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_objDec()
        self.ui.setupUi(self)

        self.train_filename = ""
        self.test_filename = ""
        self.svm_pos_foldername = ""
        self.svm_neg_foldername = ""

        self.test_GT = np.array(((79, 352, 190, 450), (163, 226, 301, 328), (245, 87, 390, 188), (354, 23, 451, 124),
        (616, 570, 782, 683), (964, 238, 1060, 330)), dtype=np.int32)
        self.train_gt = [307, 218, 628, 536]

        self.Crop_num = 200 # 裁剪的区域数量
        self.L = 300 # 区域的边长
        self.train_num = 10000
        self.lr = 0.01
        self.step = 4
        self.score_threshold = 0.7
        self.H_size = 32 # HOG特征的大小
        self.F_n = ((self.H_size // 8) ** 2) * 9 # HOG特征向量的长度
        self.nn = None
        self.recs = np.array(((100, 100), (150, 150), (200, 200)), dtype=np.float32)
        self.iou_th = 0.5
        self.detects = np.ndarray((0, 5), dtype=np.float32)

        self.flag = True

        self.connect_signal()

    def observe_L(self):
        self.L = self.ui.L.value()
        # print(self.L)

    def observe_Crop(self):
        self.Crop_num = self.ui.crop_num.value()
        # print(self.Crop_num)

    def observe_train_num(self):
        self.train_num = self.ui.train_num.value()
        # print(self.train_num)

    def observe_hog(self):
        self.H_size = self.ui.hog.value()
        self.F_n = ((self.H_size // 8) ** 2) * 9
        # print(self.H_size)

    def observe_lr(self):
        self.lr = self.ui.lr.value()
        # print(self.lr)

    def observe_step(self):
        self.step = self.ui.win_step.value()

    def observe_iou_th(self):
        self.iou_th = self.ui.iou_th.value()
        # print(self.iou_th)

    def observe_score_th(self):
        self.score_threshold = self.ui.score_thre.value()

    def observe_gpu(self):
        self.flag = self.ui.checkBox.isChecked()
        # print(self.flag)

    def show_train_image(self):
        self.train_filename = QFileDialog.getOpenFileName(self, dir="F:\\2023\\objectDetection", filter="*.jpg;*.png;*.jpeg")
        if self.train_filename[0]:
            self.train_filename = self.train_filename[0]
        self.ui.original.setPixmap(QPixmap(self.train_filename).scaled(self.ui.original.size()))


    def show_test_image(self):
        self.test_filename = QFileDialog.getOpenFileName(self, dir="F:\\2023\\objectDetection", filter="*.jpg;*.png;*.jpeg")
        if self.test_filename[0]:
            self.test_filename = self.test_filename[0]
        self.ui.dst.setPixmap(QPixmap(self.test_filename).scaled(self.ui.result.size()))

    def train(self):
        print('training...')
        img = cv2.imread(self.train_filename)
        db = func.crop_db(self.Crop_num,img,self.L,self.H_size,self.F_n,self.train_gt)

        if self.flag == False:
            self.nn = func.NN(ind=self.F_n, lr = self.lr)
            for i in range(self.train_num):
                self.nn.forward(db[:, :self.F_n])
                self.nn.train(db[:, :self.F_n], db[:, -1][..., None])

        else:
            self.nn = CNN(ind=self.F_n, w=64, w2=64, outd=1, lr=self.lr)
            # Convert the input and target to PyTorch tensors
            x = torch.tensor(db[:, :self.F_n], dtype=torch.float32)
            t = torch.tensor(db[:, -1], dtype=torch.float32).unsqueeze(1)

            # Train the network
            for i in range(self.train_num):
                self.nn.train(x, t)


        self.ui.original.setPixmap(QPixmap("./segment.jpg").scaled(self.ui.original.size()))
        print('Done')

    def detection(self):
        print('detecting...')
        img = cv2.imread(self.test_filename)
        temp = img.copy()

        self.detects = func.slide_window(img,self.nn,self.detects,self.recs,self.H_size,self.step,self.flag,self.score_threshold)
        self.detects = self.detects[func.nms(self.detects, iou_th=0.25)]

        for d in self.detects:
            v = list(map(int, d[:4]))
            cv2.rectangle(temp, (v[0], v[1]), (v[2], v[3]), (255,0,0), 1)
            cv2.putText(temp, "{:.2f}".format(d[-1]), (v[0], v[1]+9),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,0,255), 1)

        cv2.imwrite("./nms.jpg",temp)
        print('Done')

    def evaluate_result(self):
        print('begin evaluating...')
        img = cv2.imread(self.test_filename)
        Ps = func.evaluation(self.test_GT,self.detects,self.iou_th)
        func.cal_mAP(self.detects,Ps)
        func.dsiplay_final(img,self.detects,self.test_GT,Ps)
        self.ui.result.setPixmap(QPixmap("./final.jpg").scaled(self.ui.original.size()))

    def pick_svm_pos_folder(self):
        self.svm_pos_foldername = QFileDialog.getExistingDirectory(self, dir="F:\\2023\\objectDetection", caption="pick folder")
        # print(self.svm_pos_foldername)

    def pick_svm_neg_folder(self):
        self.svm_neg_foldername = QFileDialog.getExistingDirectory(self, dir="F:\\2023\\objectDetection", caption="pick folder")
        # print(self.svm_neg_foldername)

    def read_file(self):
        with open('svm_output.txt', 'r') as file:
            text = file.read()
        self.ui.svm_text.setText(text)
        file.close()


    def bt_train_svm(self):
        print('begin training svm...')
        SVM.do_training(self.svm_pos_foldername, 200, self.svm_neg_foldername, 200)
        self.read_file()
        print('Done')

    def bt_run_svm(self):
        print(self.test_filename)
        print('begin runing svm...')
        SVM.run(self.test_filename,10,150,150)
        self.ui.result.setPixmap(QPixmap(".\svm_result.jpg").scaled(self.ui.original.size()))
        print('Done')


    def show_svm_dst_image(self):
        self.test_filename = QFileDialog.getOpenFileName(self, dir="F:\\2023\\objectDetection", filter="*.jpg;*.png;*.jpeg")
        if self.test_filename[0]:
            self.test_filename = self.test_filename[0]
        self.ui.dst.setPixmap(QPixmap(self.test_filename).scaled(self.ui.original.size()))

    def pick_pretrained_dst_img(self):
        self.test_filename = QFileDialog.getOpenFileName(self, dir="F:\\2023\\objectDetection", filter="*.jpg;*.png;*.jpeg")
        if self.test_filename[0]:
            self.test_filename = self.test_filename[0]
        self.ui.dst.setPixmap(QPixmap(self.test_filename).scaled(self.ui.original.size()))

    def run_pretrained(self):
        print('begin detecting...')
        pre_trained.run(self.test_filename)
        self.ui.result.setPixmap(QPixmap(".\pre-trained-result.jpg").scaled(self.ui.original.size()))
        print('Done')


    def connect_signal(self):
        self.ui.pick.clicked.connect(self.show_train_image)
        self.ui.pick2.clicked.connect(self.show_test_image)
        self.ui.train.clicked.connect(self.train)
        self.ui.detect.clicked.connect(self.detection)
        self.ui.evaluation.clicked.connect(self.evaluate_result)

        self.ui.L.valueChanged.connect(self.observe_L)
        self.ui.lr.valueChanged.connect(self.observe_lr)
        self.ui.train_num.valueChanged.connect(self.observe_train_num)
        self.ui.iou_th.valueChanged.connect(self.observe_iou_th)
        self.ui.crop_num.valueChanged.connect(self.observe_Crop)
        self.ui.hog.valueChanged.connect(self.observe_hog)
        self.ui.win_step.valueChanged.connect(self.observe_step)
        self.ui.checkBox.clicked.connect(self.observe_gpu)
        self.ui.score_thre.valueChanged.connect(self.observe_score_th)

        self.ui.svm_pos_folder.clicked.connect(self.pick_svm_pos_folder)
        self.ui.svm_neg_folder.clicked.connect(self.pick_svm_neg_folder)
        self.ui.svm_dst.clicked.connect(self.show_svm_dst_image)
        self.ui.train_svm.clicked.connect(self.bt_train_svm)
        self.ui.run_svm.clicked.connect(self.bt_run_svm)

        self.ui.pick_pretrained_img.clicked.connect(self.pick_pretrained_dst_img)
        self.ui.run_model.clicked.connect(self.run_pretrained)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = objDec()
    widget.show()
    sys.exit(app.exec())
