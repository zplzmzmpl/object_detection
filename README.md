# object_detection
using hog+nn and svm and pretrained model to detect object(plane as example)
使用pyside6界面模块以及Qt6 designer进行搭建ui文件。将ui文件转为py脚本，在主函数中继承ui类，连接功能槽函数。本工程全部使用python语言编程实现。
 
测试数据
 
检测目标图像
	HOG+NN模块
用户参数设置：
	Crop_num为裁剪的区域数量，用于创建训练数据集
	Seg_size 为裁剪区域的边长
	train_num为训练次数
	learning_rate为神经网络学习率
	step为滑动窗口步长
	score_threshold为目标得分阈值
	H_size为HOG特征的大小
	F_n为HOG特征向量的长度
	Iou_threshold为矩形交并比阈值
	Pytorch为复选框表示是否启用pytorch框架
用户可按需求设定参数，若无需要保持默认参数即可。
 
Fig 1. 选取训练图像与目标影像
设计基于pytorch框架的简单的卷积神经网络代替原结构，包括两层全连接层和一个输出层，使用Sigmoid和ReLu为激活函数，交叉熵损失函数，优化器为SGD随机梯度下降。
 
Fig 2. 使用pytorch框架重写神经网络结构
蓝色为目标正数据集，红色为非目标负数据集
 
Fig 3. 创建数据集结果
 
Fig 4. NN检测结果
由下图可知，召回率为0.83，较高；但准确率仅为0.29，低；F得分以及mAP值分别为0.434与0.228，精度较低。在影像右上区域存在大量误分，但目标均有被检测到。
 
Fig 5. nms结果
 
Fig 6. 精度验证
	SVM+HOG模块
准备正负样本训练集，尺寸均为64x128，训练数量均为200
 
Fig 7. 正样本数据集
 
Fig 8. 负样本数据集
 
Fig 9. 选取目标影像
输入正负样本集后，点击训练输出权重数据。
 
Fig 10. 训练结果weight权重数据
由下图可知，召回率为0.5，准确率为0.428
 
Fig 11. SVM+HOG结果
	预训练模型模块
 

Fig 12. 预训练模型选取目标影像
由结果可知，召回率为0.8333
 
 
Fig 13. 运行结果
