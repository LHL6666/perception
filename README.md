## 该项目是ICRA2020 AI Challenge ChongShi战队的感知功能代码说明 --version1.0.1
## **1. 软件功能介绍**  
感知功能实现了敌我机器人识别、敌方机器人装甲板识别、敌方机器人尾灯姿态识别、哨岗识别敌方机器人及其装甲板、哨岗识别敌方机器人位置坐标。
#### **机器人和装甲板识别算法框架：**    
使用了深度学习目标检测算法：一共尝试了**yolov4, yolov4-tiny, "yolov5", AlexeyAB发行的tensorRT加速版yolov4-tiny**框架，其中tensorRT加持的yolov4-tiny算法在我们的机载PC（仅为ARM处理器）上识别速度能够高达近乎150帧，能够在三米内稳定识别装甲板和尾灯，7米之内稳定识别机器人（有一定的防遮挡能力），但是由于对装甲板的识别距离太近，限制了日后决策组发展，因此最后采取折中的方案，最终在哨岗视觉和机器人视觉处理上使用了来自ultralytics公司的"yolov5"框架，该框架在调节模型大小和优化器等参数后取得较好的效果，能够在六米内稳定识别装甲板和尾灯，识别机器人的准确度也很高。 
主要功能对比如下（以最后模型为准）：   
______________________________________________________________________________________________________   
#####│                         ultralytics yolov5：           │         AlexeyAB版 yolov4-tiny        │ 
                                                              
6米距离识别机器人:                      94%                    |                    80%                │
6米距离识别装甲板:                      72%                    │                    NO                 │
6米稳定识别时帧率:              30~85fps(size最大最小)                     50~150fps(size最大最小)        
——————————————————————————————————————————————————————————————————————————————————————————————————————

## **2. 软件效果展示**  
<p align="center"><img style="display: block; margin: 0 auto;" src="Wust_perception/Innocent_Bird-master/image/哨岗视觉识别.gif" width="80%" alt="" /></p>   
<p align="center">哨岗视觉识别</p>  
<p align="center"><img style="display: block; margin: 0 auto;" src="image/检测.gif" width="80%" alt="" /></p>   
<p align="center">机器人目标检测</p>  
<p align="center"><img style="display: block; margin: 0 auto;" src="image/位姿识别.gif" width="80%" alt="" /></p>   
<p align="center">基于先验信息和模板匹配的位姿识别</p>   
![image](https://github.com/ZhengYaWei1992/ZWProgressView/blob/master/Untitled3.gif)

## **3. 依赖工具，软、硬件环境**
####**软件部分：**   
系统版本：Ubuntu18.04    
机载平台：Jetpack 4.4，tensorRT, PyTorch, TensorFlow, 具体版本要求如下：

CUDA 10.2   
PyTorch1.6   
PyYAML>=5.3     
scipy>=1.4.1    
tqdm>=4.41.0    
numpy>=1.18.5   
TensorFlow>=2.2   
matplotlib>=3.2.2     
torchvision>=0.7.0      
OpenCV-python>=4.1.2    
tensorRT: jetpack 4.4刷机时安装即可    

####**硬件部分：**   
机载平台： Jetson AGX Xavier
哨岗电脑： 小米Pro  Intel i7-8550U @1.8G + GeForce MX150      
单目摄像头：威鑫视界WX605摄像头，镜头150°，焦距2.45mm，分辨率1280*720，帧率 120   
深度相机： Intel D435i深度相机，RGB：1920x1080, 30FPS, 深度图像：最高1280x720主动立体深度图，FPS: 90 max


# **4. 编译、安装方式**
### **linux下**：
① 克隆代码至本地仓库：git clone https://github.com/LHL6666/perception.git Wust_Perception   
② 添加该工程路径到python的搜索路径，使python解释器能够找到models,runs,utils文件夹下的python文件，或者直接将这三个文件夹复制到
python site-packages里面，例如~/.local/lib/python3.6/site-packages/
③ 修改Innocent_Bird.py文件，更改模型搜索路径，更改摄像头的编号后，打开终端cd到该工程根目录下面，直接运行 python3 Innocent_Bird.py     
### **window10下**：
① 下载工程：     
② 使用pycharm等软件打开工程文件    
③ 直接运行Innocent_Bird.py      

# **5. 软件使用说明** 
## ***A. 深度学习目标检测算法***   
深度学习目标检测算法用来实现敌方机器人的识别以及哨岗识别，使用python语言编写，算法代码位于RobDet文件夹下。本项目共提供两组模型：

- 部署在机器人平台的检测模型——TinyYOLOv3，该模型是由我们组对Darknet版本的TinyYOLOv3的PyTorch复现，添加了SPP模块。
- 部署在哨岗平台的检测模型——SlimYOLOv2，该模型是由我们组对Darknet版本的YOLOv2的PyTorch复现，并且做了轻量化处理和添加SPP模块。

其中，我们将YOLOv2中的backbone网络——Darknet19替换为由我们自主设计的轻量级网络Darknet_tiny，该网络在ImageNet数据集上进行预训练，在val上获得
top1精度63.5和top5精度85.06。为了提高网络的性能，同时不引入过多的计算量，在detection head部分添加了SPP模块。

## 数据集
哨岗训练用的数据集是场地的鸟瞰图，我在Innocent_Bird.py文件里面加入了采集数据集的功能，在

## 模型
已训练好的模型请使用下面的百度网盘链接下载（14MB左右）：   

### 哨岗检测模型
链接：https://pan.baidu.com/s/15dIvgZN781N9q14gnpFfZw      

提取码：wifw       


### 机载检测模型
链接：https://pan.baidu.com/s/1HHIdqMT0tnO45W5gij9R0A      

提取码：5twr    

### yolov5s权重
链接: https://pan.baidu.com/s/1Ge1--weNoh_KgB2xFRRQ2A  	
提取码: m0hl


# **6. 文件目录结构及文件用途说明**   
 ```
Innocent_Bird-master.
├── models
│   ├── common.py // 包含了yolov5s、yolov5m、yolov5l和yolov5x模型通用的模块，还包括了SPP等结构
│   ├── export.py // 将训练好的.pt模型转换成onnx和TorchScript格式，减少由于训练模型时保存的设备和时间参数等，增加通用性以及可用于tensorRT加速处理
│   ├── experimental.py // 包含实验模块还有加载训练好的模型函数，比较新颖的MixConv2d混合神经网络模块都在里面有体现
│   └── yolo.py  //模型文件，包含了用来解析输入的yolov5s.yaml参数网络的功能
│
├── utils
│   ├── activations.py
│   ├── datasets.py
|   ├── general.py
│   ├── googles_utils.py
│   ├── torch_utils.py
│   └── utils.py
│ 
├── Camera_Calibration.py
├── Camera_Calibration.py
├── Innocent_Bird.py
├── requirements.txt
├── test.py
└── train.py // 说明文件
```  

# **7. 原理介绍与理论支持分析**   
## 1. 机器人与装甲板识别及哨岗识别  
&emsp;&emsp;为实现自动识别视野中敌方机器人与装甲板，本方案中使用基于深度学习的目标检测算法。现有的目标检测算法，诸如Faster-RCNN、SSD、YOLO等先进的检测算法模型，已在公开数据集PASCAL VOC与MSCOCO上均表现出了优异的性能和识别精度，但是这些算法模型过大，依托于TITAN X这一类的高性能、高成本的GPU来保证目标检测的实时性。考虑到比赛中所使用的AI机器人无法承载高性能GPU及其相关硬件配置，不能为以上的模型提供足够的算力支持，我们队伍改进YOLO-v1算法，将其轻量化，使其能够在较低性能的GTX-1060-modile GPU上具有40+FPS的快速检测能力。
&emsp;&emsp;目标检测网络从功能上，可以分为两部分：主干网络（Backbone）和检测网络（Detection head）。前者负责从输入图像中提取感兴趣的目标特征，后者利用这些特征进行检测，识别出目标的类别并定位。主干网络这一部分，本方案使用轻量网络ResNet-18，其网络结构如图一所示。ResNet-18包含的参数量为11.69M，浮点计算量为1.58GFLOPs，其较少的参数量和计算量，适用于低性能GPU。  
<p align="center"><img style="display: block; margin: 0 auto;" src="PIC/resnet18网络结构.png" width="100%" alt="" /></p>  
<p align="center">图7-1 ResNet-18网络结构</p>  
&emsp;&emsp;检测网络中，本方案使用简单的网络结构，如图二所示，图中展示的Block模块由一层1×1的reduction卷积层接3×3卷积层构成。这样简易的结构既能增加网络结构的深度，同时还不会带来过多的参数和计算量，有助于将模型部署在低性能GPU上。  
<p align="center"><img style="display: block; margin: 0 auto;" src="PIC/block结构.png" width="25%" alt="" /></p>  
<p align="center">图7-2 Block结构</p>  
&emsp;&emsp;结合以上两部分，便可以搭建本方案中所使用的目标检测网络RobDet，其整体结构如图三所示。给定输入图片，主干网络首先在图像中提取特征。这里，为了加快网络的收敛速度，本方案使用ResNet-18在ImageNet上已经预训练好的模型，将其加载到主干网络中。主干网路在提取出特征后，输出特征图，并由后续的检测网络实现对视野中的机器人及其装甲板的识别和定位。  
<p align="center"><img style="display: block; margin: 0 auto;" src="PIC/RobDet.png" width="80%" alt="" /></p>  
<p align="center">图7-3 Block结构</p>  
&emsp;&emsp;为了完成此目标，本方案中采用dahua-A5131CU210单目相机采集机器人及其装甲板的训练集图像。共采集不同角度和距离下的机器人图片1784张，部分结果如图四所示。训练模型所需的实验设备为Intel i9-9940CPU和一块TITAN RTX显卡。训练中，batch大小设置为64，初始学习率为0.001，训练90个epoch，每30个epoch，学习率降低10%。训练过程中的损失函数曲线在图四中给出。  
<p align="center"><img style="display: block; margin: 0 auto;" src="PIC/训练曲线.png" width="80%" alt="" /></p>  
<p align="center">图7-4 训练曲线 左图：类别识别损失函数曲线 右图：定位损失函数曲线</p>   
<p align="center"><img style="display: block; margin: 0 auto;" src="PIC/识别效果.png" width="80%" alt="" /></p>  
<p align="center">图7-5 部分RobDet检测结果的可视化</p>  
<p align="center"><img style="display: block; margin: 0 auto;" src="PIC/哨岗.png" width="100%" alt="" />  
<p align="center">图7-6 哨岗识别效果</p>  

## 2. 机器人姿态估计  
  

### 3. 机器人运动预测
初步测试了KCF、MOSSE和CSRT等传统跟踪算法，发现MOSSE算法对该哨岗视觉算法最合适的，在机器人被遮挡大部分时仍能够正常跟踪不容易丢失目标，
KCF虽然能够达到300多帧的跟踪速度，但是精度和抗干扰性都不是很好，MOSSE在我的测试过程中保持了120帧左右的跟踪速度，精度和抗干扰性好很
多。但是由于第一届参加比赛还没有得到固定场地，还没录视频就被迫更换场地，没法固定哨岗相机不满足测试条件了

# **8. 数据流图及软件框图**  
<p align="center"><img style="display: block; margin: 0 auto;" src="PIC/软件框图.png" width="30%" alt="" /></p>  
<p align="center">图8-1 软件框图</p>  
<p align="center"><img style="display: block; margin: 0 auto;" src="PIC/总框图.png" width="80%" alt="" /></p>  
<p align="center">图8-2 总数据流框图</p>  


### Reference   
https://docs.opencv.org/master/d9/df8/tutorial_root.html    
https://github.com/ultralytics/yolov5   
