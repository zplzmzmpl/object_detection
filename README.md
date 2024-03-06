# object_detection

*using hog+nn and svm and pretrained model to detect object(plane as example)*

**使用pyside6界面模块以及Qt6 designer进行搭建ui文件。将ui文件转为py脚本，在主函数中继承ui类，连接功能槽函数。本工程全部使用python语言编程实现。**
 
**HOG++NN模块训练数据**

![image](https://github.com/zplzmzmpl/object_detection/assets/121420991/90e2a956-0249-4580-80db-a68a935b163b)

**检测目标图像**

![image](https://github.com/zplzmzmpl/object_detection/assets/121420991/93eea72a-b351-4473-87ef-5e0fc3d0fe59)

* ***HOG+NN模块***

用户参数设置：
	*Crop_num为裁剪的区域数量，用于创建训练数据集
 * Seg_size 为裁剪区域的边长
 * train_num为训练次数
 * learning_rate为神经网络学习率
 * step为滑动窗口步长
 * score_threshold为目标得分阈值
 * H_size为HOG特征的大小
 * F_n为HOG特征向量的长度
 * Iou_threshold为矩形交并比阈值
 * Pytorch为复选框表示是否启用pytorch框架

**用户可按需求设定参数，若无需要保持默认参数即可。**
 
**Fig 1. 选取训练图像与目标影像**

![image](https://github.com/zplzmzmpl/object_detection/assets/121420991/761b38a7-308e-42c6-8583-e0eb26db0fc6)

设计基于pytorch框架的简单的卷积神经网络代替原结构，包括两层全连接层和一个输出层，使用Sigmoid和ReLu为激活函数，交叉熵损失函数，优化器为SGD随机梯度下降。

**Fig 2. 使用pytorch框架重写神经网络结构**

![image](https://github.com/zplzmzmpl/object_detection/assets/121420991/2eeb60ef-409a-4818-83e3-ba504bf7fbc9)

**Fig 3. 上图创建数据集结果（蓝色为目标正数据集，红色为非目标负数据集）**

![image](https://github.com/zplzmzmpl/object_detection/assets/121420991/a4d16533-e788-4fe1-b8aa-bf7a31d3a6c8)

**Fig 4. NN检测结果**

![image](https://github.com/zplzmzmpl/object_detection/assets/121420991/3780a9a5-ce80-4118-8361-5265e32c0cc8)

由上图可知，召回率为0.83，较高；但准确率仅为0.29，低；F得分以及mAP值分别为0.434与0.228，精度较低。在影像右上区域存在大量误分，但目标均有被检测到。

**Fig 5. nms结果**

![image](https://github.com/zplzmzmpl/object_detection/assets/121420991/15d3e01e-514d-4610-a8d1-c204a615e46f)

**Fig 6. 精度验证**

![image](https://github.com/zplzmzmpl/object_detection/assets/121420991/7e54b02d-21cc-421d-a107-a8516406557d)


* ***SVM+HOG模块***

准备正负样本训练集，尺寸均为64x128，训练数量均为200

**Fig 7. 正样本数据集**

![image](https://github.com/zplzmzmpl/object_detection/assets/121420991/433aafe5-2ad3-422e-b9e7-adc6da8d2a8e)

**Fig 8. 负样本数据集**

![image](https://github.com/zplzmzmpl/object_detection/assets/121420991/b1a0dfe1-2ef0-4cbc-8b37-285129f5a9d2)

**Fig 9. 选取目标影像**

![image](https://github.com/zplzmzmpl/object_detection/assets/121420991/385f39f7-3c91-4359-aaaa-a1b10e06c402)

输入正负样本集后，点击训练输出权重数据。

Fig 10. 训练结果weight权重数据

![image](https://github.com/zplzmzmpl/object_detection/assets/121420991/081a627f-4e70-4435-b39f-5c74d13f52e7)

**Fig 11. SVM+HOG结果（由下图可知，召回率为0.5，准确率为0.428）**

![image](https://github.com/zplzmzmpl/object_detection/assets/121420991/f4e961d5-adb2-480e-9c2d-39168710b61a)


* ***预训练模型模块***

**Fig 12. 预训练模型选取目标影像**

![image](https://github.com/zplzmzmpl/object_detection/assets/121420991/f4363c1a-20a6-48d0-afec-db0973489e8d)


由结果可知，召回率为0.8333

![image](https://github.com/zplzmzmpl/object_detection/assets/121420991/37670bd6-44bf-4e34-87b8-9c2a0bab3f79)

![image](https://github.com/zplzmzmpl/object_detection/assets/121420991/0f404d94-c375-4992-ac5d-137690c5a8c7)

---

# 【源程序清单及核心代码】

/hog_nn.py/-------iou() function # 计算iou精度

		  -------hog() function # 计算hog特征
				
		  -------resize() function # 图像裁剪
				
		  -------crop_db() function # 创建训练数据集
				
		  -------slid_window() function # 滑动窗口
				
		  -------evaluation() function # 评估函数
				
		  -------nms() function # 非极值抑制
				
/CNN.py/-------class cnn:

------__init__() function # 初始化参数

------forward() function # 前向传播

				------train() function # 训练函数
				
/yolo_pretrained.py/--------run_mode() function # 调用预训练模型

/HOG.py/------ class Hog_descriptor():

			 -------extract() function # 主功能函数，计算图像的HOG描述符
				
			 -------global_gradient() function # 辅助函数，使用sobel算子计算x、y的梯度和方向
				
			 --------- cell_gradient() function # 辅助函数，为一个cell单元构建梯度方向直方图
				
			 --------- gradient_image() function # 辅助函数，将梯度直方图转化为特征图像
				
/SVM.py/------- kernelTrans() function # 计算核函数值，将样本映射到高维空间

		-------class optStruct() # 求解核函数值的类
		
		------- innerL() function # SVM优化alpha函数
		
		-------smoP() function # 调用SMO函数并计算权重
		
		-------SMO() function # SMO算法优化SVM
		
		-------LoadDataset() function # 加载数据集
		
		-------do_training() function # 训练函数
		
		-------run() function # 运行函数
