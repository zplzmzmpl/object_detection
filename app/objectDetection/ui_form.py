# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'form.ui'
##
## Created by: Qt User Interface Compiler version 6.6.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QCheckBox, QDoubleSpinBox, QFrame,
    QGridLayout, QHBoxLayout, QLabel, QPushButton,
    QSizePolicy, QSpinBox, QSplitter, QTabWidget,
    QTextBrowser, QVBoxLayout, QWidget)

class Ui_objDec(object):
    def setupUi(self, objDec):
        if not objDec.objectName():
            objDec.setObjectName(u"objDec")
        objDec.resize(1139, 515)
        self.gridLayout = QGridLayout(objDec)
        self.gridLayout.setObjectName(u"gridLayout")
        self.splitter_2 = QSplitter(objDec)
        self.splitter_2.setObjectName(u"splitter_2")
        self.splitter_2.setOrientation(Qt.Horizontal)
        self.tabWidget = QTabWidget(self.splitter_2)
        self.tabWidget.setObjectName(u"tabWidget")
        self.nn = QWidget()
        self.nn.setObjectName(u"nn")
        self.gridLayout_3 = QGridLayout(self.nn)
        self.gridLayout_3.setObjectName(u"gridLayout_3")
        self.gridLayout_2 = QGridLayout()
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.original = QLabel(self.nn)
        self.original.setObjectName(u"original")
        self.original.setScaledContents(True)
        self.original.setAlignment(Qt.AlignCenter)

        self.gridLayout_2.addWidget(self.original, 0, 1, 1, 1)

        self.verticalLayout_2 = QVBoxLayout()
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.checkBox = QCheckBox(self.nn)
        self.checkBox.setObjectName(u"checkBox")

        self.verticalLayout_2.addWidget(self.checkBox)

        self.gt_tl_x = QSpinBox(self.nn)
        self.gt_tl_x.setObjectName(u"gt_tl_x")
        self.gt_tl_x.setMinimum(0)
        self.gt_tl_x.setMaximum(5000)
        self.gt_tl_x.setSingleStep(1)
        self.gt_tl_x.setValue(0)

        self.verticalLayout_2.addWidget(self.gt_tl_x)

        self.gt_tl_y = QSpinBox(self.nn)
        self.gt_tl_y.setObjectName(u"gt_tl_y")
        self.gt_tl_y.setMinimum(0)
        self.gt_tl_y.setMaximum(5000)
        self.gt_tl_y.setSingleStep(1)
        self.gt_tl_y.setValue(0)

        self.verticalLayout_2.addWidget(self.gt_tl_y)

        self.gt_br_x = QSpinBox(self.nn)
        self.gt_br_x.setObjectName(u"gt_br_x")
        self.gt_br_x.setMinimum(0)
        self.gt_br_x.setMaximum(5000)
        self.gt_br_x.setSingleStep(1)
        self.gt_br_x.setValue(0)

        self.verticalLayout_2.addWidget(self.gt_br_x)

        self.gt_br_y = QSpinBox(self.nn)
        self.gt_br_y.setObjectName(u"gt_br_y")
        self.gt_br_y.setMinimum(0)
        self.gt_br_y.setMaximum(5000)
        self.gt_br_y.setSingleStep(1)
        self.gt_br_y.setValue(0)

        self.verticalLayout_2.addWidget(self.gt_br_y)

        self.L = QSpinBox(self.nn)
        self.L.setObjectName(u"L")
        self.L.setMinimum(10)
        self.L.setMaximum(5000)
        self.L.setSingleStep(10)
        self.L.setValue(300)

        self.verticalLayout_2.addWidget(self.L)

        self.win_step = QSpinBox(self.nn)
        self.win_step.setObjectName(u"win_step")
        self.win_step.setMinimum(1)
        self.win_step.setMaximum(1000)
        self.win_step.setSingleStep(1)
        self.win_step.setValue(4)

        self.verticalLayout_2.addWidget(self.win_step)

        self.hog = QSpinBox(self.nn)
        self.hog.setObjectName(u"hog")
        self.hog.setMaximum(256)
        self.hog.setValue(32)

        self.verticalLayout_2.addWidget(self.hog)

        self.score_thre = QDoubleSpinBox(self.nn)
        self.score_thre.setObjectName(u"score_thre")
        self.score_thre.setMaximum(1.000000000000000)
        self.score_thre.setSingleStep(0.010000000000000)
        self.score_thre.setValue(0.700000000000000)

        self.verticalLayout_2.addWidget(self.score_thre)

        self.train_num = QSpinBox(self.nn)
        self.train_num.setObjectName(u"train_num")
        self.train_num.setMinimum(200)
        self.train_num.setMaximum(100000)
        self.train_num.setSingleStep(100)
        self.train_num.setValue(10000)

        self.verticalLayout_2.addWidget(self.train_num)

        self.lr = QDoubleSpinBox(self.nn)
        self.lr.setObjectName(u"lr")
        self.lr.setMaximum(1.000000000000000)
        self.lr.setSingleStep(0.010000000000000)
        self.lr.setValue(0.010000000000000)

        self.verticalLayout_2.addWidget(self.lr)

        self.crop_num = QSpinBox(self.nn)
        self.crop_num.setObjectName(u"crop_num")
        self.crop_num.setMinimum(10)
        self.crop_num.setMaximum(1000)
        self.crop_num.setSingleStep(10)
        self.crop_num.setValue(200)

        self.verticalLayout_2.addWidget(self.crop_num)

        self.iou_th = QDoubleSpinBox(self.nn)
        self.iou_th.setObjectName(u"iou_th")
        self.iou_th.setMaximum(1.000000000000000)
        self.iou_th.setSingleStep(0.010000000000000)
        self.iou_th.setValue(0.500000000000000)

        self.verticalLayout_2.addWidget(self.iou_th)


        self.gridLayout_2.addLayout(self.verticalLayout_2, 0, 0, 1, 1)

        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.pick = QPushButton(self.nn)
        self.pick.setObjectName(u"pick")

        self.horizontalLayout.addWidget(self.pick)

        self.train = QPushButton(self.nn)
        self.train.setObjectName(u"train")

        self.horizontalLayout.addWidget(self.train)

        self.pick2 = QPushButton(self.nn)
        self.pick2.setObjectName(u"pick2")

        self.horizontalLayout.addWidget(self.pick2)

        self.detect = QPushButton(self.nn)
        self.detect.setObjectName(u"detect")

        self.horizontalLayout.addWidget(self.detect)

        self.evaluation = QPushButton(self.nn)
        self.evaluation.setObjectName(u"evaluation")

        self.horizontalLayout.addWidget(self.evaluation)


        self.gridLayout_2.addLayout(self.horizontalLayout, 1, 0, 1, 2)


        self.gridLayout_3.addLayout(self.gridLayout_2, 0, 0, 1, 1)

        self.tabWidget.addTab(self.nn, "")
        self.svm = QWidget()
        self.svm.setObjectName(u"svm")
        self.gridLayout_4 = QGridLayout(self.svm)
        self.gridLayout_4.setObjectName(u"gridLayout_4")
        self.svm_pos_folder = QPushButton(self.svm)
        self.svm_pos_folder.setObjectName(u"svm_pos_folder")

        self.gridLayout_4.addWidget(self.svm_pos_folder, 0, 0, 1, 1)

        self.svm_neg_folder = QPushButton(self.svm)
        self.svm_neg_folder.setObjectName(u"svm_neg_folder")

        self.gridLayout_4.addWidget(self.svm_neg_folder, 0, 1, 1, 1)

        self.train_svm = QPushButton(self.svm)
        self.train_svm.setObjectName(u"train_svm")

        self.gridLayout_4.addWidget(self.train_svm, 0, 2, 1, 1)

        self.svm_dst = QPushButton(self.svm)
        self.svm_dst.setObjectName(u"svm_dst")

        self.gridLayout_4.addWidget(self.svm_dst, 0, 3, 1, 1)

        self.run_svm = QPushButton(self.svm)
        self.run_svm.setObjectName(u"run_svm")

        self.gridLayout_4.addWidget(self.run_svm, 0, 4, 1, 1)

        self.svm_text = QTextBrowser(self.svm)
        self.svm_text.setObjectName(u"svm_text")

        self.gridLayout_4.addWidget(self.svm_text, 1, 0, 1, 5)

        self.tabWidget.addTab(self.svm, "")
        self.pre = QWidget()
        self.pre.setObjectName(u"pre")
        self.gridLayout_5 = QGridLayout(self.pre)
        self.gridLayout_5.setObjectName(u"gridLayout_5")
        self.pick_pretrained_img = QPushButton(self.pre)
        self.pick_pretrained_img.setObjectName(u"pick_pretrained_img")

        self.gridLayout_5.addWidget(self.pick_pretrained_img, 0, 0, 1, 1)

        self.run_model = QPushButton(self.pre)
        self.run_model.setObjectName(u"run_model")

        self.gridLayout_5.addWidget(self.run_model, 0, 1, 1, 1)

        self.pretrained_text = QTextBrowser(self.pre)
        self.pretrained_text.setObjectName(u"pretrained_text")

        self.gridLayout_5.addWidget(self.pretrained_text, 1, 0, 1, 2)

        self.tabWidget.addTab(self.pre, "")
        self.splitter_2.addWidget(self.tabWidget)
        self.splitter = QSplitter(self.splitter_2)
        self.splitter.setObjectName(u"splitter")
        self.splitter.setOrientation(Qt.Vertical)
        self.dst = QLabel(self.splitter)
        self.dst.setObjectName(u"dst")
        self.dst.setScaledContents(True)
        self.dst.setAlignment(Qt.AlignCenter)
        self.splitter.addWidget(self.dst)
        self.hline = QFrame(self.splitter)
        self.hline.setObjectName(u"hline")
        self.hline.setFrameShape(QFrame.HLine)
        self.hline.setFrameShadow(QFrame.Sunken)
        self.splitter.addWidget(self.hline)
        self.result = QLabel(self.splitter)
        self.result.setObjectName(u"result")
        self.result.setScaledContents(True)
        self.result.setAlignment(Qt.AlignCenter)
        self.splitter.addWidget(self.result)
        self.splitter_2.addWidget(self.splitter)

        self.gridLayout.addWidget(self.splitter_2, 0, 0, 1, 1)


        self.retranslateUi(objDec)

        self.tabWidget.setCurrentIndex(2)


        QMetaObject.connectSlotsByName(objDec)
    # setupUi

    def retranslateUi(self, objDec):
        objDec.setWindowTitle(QCoreApplication.translate("objDec", u"objDec", None))
        self.original.setText(QCoreApplication.translate("objDec", u"original", None))
        self.checkBox.setText(QCoreApplication.translate("objDec", u"pytorch", None))
        self.gt_tl_x.setSuffix("")
        self.gt_tl_x.setPrefix(QCoreApplication.translate("objDec", u"gt top left x: ", None))
        self.gt_tl_y.setSuffix("")
        self.gt_tl_y.setPrefix(QCoreApplication.translate("objDec", u"gt top left y: ", None))
        self.gt_br_x.setSuffix("")
        self.gt_br_x.setPrefix(QCoreApplication.translate("objDec", u"gt bottom right x: ", None))
        self.gt_br_y.setSuffix("")
        self.gt_br_y.setPrefix(QCoreApplication.translate("objDec", u"gt bottom right y: ", None))
        self.L.setSuffix("")
        self.L.setPrefix(QCoreApplication.translate("objDec", u"seg size: ", None))
        self.win_step.setSuffix("")
        self.win_step.setPrefix(QCoreApplication.translate("objDec", u"step: ", None))
        self.hog.setPrefix(QCoreApplication.translate("objDec", u"HOG size: ", None))
        self.score_thre.setPrefix(QCoreApplication.translate("objDec", u"score threshold: ", None))
        self.train_num.setPrefix(QCoreApplication.translate("objDec", u"training num: ", None))
        self.lr.setPrefix(QCoreApplication.translate("objDec", u"learning rate: ", None))
        self.crop_num.setPrefix(QCoreApplication.translate("objDec", u"crop num: ", None))
        self.iou_th.setPrefix(QCoreApplication.translate("objDec", u"iou threshold: ", None))
        self.pick.setText(QCoreApplication.translate("objDec", u"pick training", None))
        self.train.setText(QCoreApplication.translate("objDec", u"train", None))
        self.pick2.setText(QCoreApplication.translate("objDec", u"pick destination", None))
        self.detect.setText(QCoreApplication.translate("objDec", u"detect", None))
        self.evaluation.setText(QCoreApplication.translate("objDec", u"evaluate", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.nn), QCoreApplication.translate("objDec", u"HOG+NN", None))
        self.svm_pos_folder.setText(QCoreApplication.translate("objDec", u"pick train pos folder", None))
        self.svm_neg_folder.setText(QCoreApplication.translate("objDec", u"pick train neg folder", None))
        self.train_svm.setText(QCoreApplication.translate("objDec", u"training", None))
        self.svm_dst.setText(QCoreApplication.translate("objDec", u"pick destinate image", None))
        self.run_svm.setText(QCoreApplication.translate("objDec", u"run svm", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.svm), QCoreApplication.translate("objDec", u"HOG+SVM", None))
        self.pick_pretrained_img.setText(QCoreApplication.translate("objDec", u"pick destinate image", None))
        self.run_model.setText(QCoreApplication.translate("objDec", u"runing model", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.pre), QCoreApplication.translate("objDec", u"PRE-TRAINED", None))
        self.dst.setText(QCoreApplication.translate("objDec", u"destination", None))
        self.result.setText(QCoreApplication.translate("objDec", u"result", None))
    # retranslateUi

