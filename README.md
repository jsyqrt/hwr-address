# hwr-address
基于深度学习的手写汉字地址识别
1. 汉字单字识别
	* 使用 8 层CNN和 2 层全连接识别 3755 个 GB2312 一级汉字 [1]
	* 使用 2 层CNN和 1 层全连接识别 15 的中文地址关键字， ```省,市,县,区,乡,镇,村,巷,弄,路, 街, 社, 组, 队, 州```
	* 使用 HCL2000 中文汉字手写数据库，对每个汉字 包含 700 个训练样本和 300 个测试样本 [2]
2. 地址树构建
	* 使用 民政部 《2013年中华人民共和国县以下行政区划代码》作为知识库， 构建形如
		```
		中国
			上海市
				浦东新区
					张江镇
				黄浦区
				...
			...
		```
		的地址树，用于验证识别的汉字。
3. 综合识别
	* 输入： 单行图片格式中文手写汉字地址
	* 输出： 识别结果(中文字符串)
4. 模型，框架，工具
	* python语言
	* tensorflow，keras两种实现，相较之下，keras的封装实现效率更高
	* 目录
	```	
	/src 			# source code
	/data/ 			# dataset and train result
		address/	# address tree
		hcl/		# HCL2000 数据库
		result/		# train result
		sample/		# test samples，jpg pictures
	```
5. 参考文献
	1. Zhang X Y, Bengio Y, Liu C L. Online and Offline Handwritten Chinese Character Recognition: A Comprehensive Study and New Benchmark[J]. Pattern Recognition, 2016, 61(61):348-360.
	2. Zhang H, Guo J, Chen G, et al. HCL2000 - A Large-scale Handwritten Chinese Character Database for Handwritten Character Recognition[C]// International Conference on Document Analysis and Recognition. IEEE Computer Society, 2009:286-290.
6. 工具的安装和使用
	* tensorflow, Ubuntu可以直接通过``` sudo apt-get install tensorflow ```安装, GPU版本及cuda的安装参见 [这里](http://blog.csdn.net/zhaoyu106/article/details/52793183)
	* keras, keras是基于 tensorflow 或 theano 的对深度学习模型的更高一层封装，相对 tensorflow 直接实现效率更高
	```
	# keras依赖库
	sudo apt-get install liblapcak-dev, gfortran, scipy, cython, libhdf5-dev, h5py
	sudo apt-get install keras
	```
	* 注意，在 keras 中如果以 tensorflow 作为后端，数据输入格式为 (channel, row, col)时，需要修改.keras目录下的json配置文件，将 image_dim_order 改为 'th'(theano模式，即Numpy数组的默认模式)。