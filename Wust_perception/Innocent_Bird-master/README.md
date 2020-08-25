## copyright@LHL of Wust
## **1. 软件功能介绍**  
哨岗感知功能实现了敌我机器人识别、敌方机器人装甲板识别、敌方机器人位置坐标识别。
#### **机器人和装甲板识别算法框架：**    
使用了深度学习目标检测算法：一共尝试了**yolov4, yolov4-tiny, "yolov5", AlexeyAB发行的tensorRT加速版yolov4-tiny**框架，其中tensorRT加持的yolov4-tiny算法在我们的机载PC（仅为ARM处理器）上识别速度能够高达近乎150帧，能够在三米内稳定识别装甲板和尾灯，7米之内稳定识别机器人（有一定的防遮挡能力），但是由于对装甲板的识别距离太近，限制了日后决策组发展，因此最后采取折中的方案，最终在哨岗视觉和机器人视觉处理上使用了来自ultralytics公司的"yolov5"框架，该框架在调节模型大小和优化器等参数后取得较好的效果，能够在六米内稳定识别装甲板和尾灯，识别机器人的准确度也很高 
主要功能对比如下（以最后模型为准）：   

|                |ultralytics yolov5 |AlexeyAB版 yolov4-tiny                        |
|----------------|-------------------------------|-----------------------------|
|6米距离识别机器人  |  94%   |  80%  |
|6米距离识别装甲板  |  72%   |  NO   |
|稳定识别时的FPS(size不同)  |30-85   |   50-150|


## **2. 软件效果展示**   
#### **若无法加载图像，建议下载工程后到Innocent_Bird-master/images/文件夹下打开** 
<p align="center"><img style="display: block; margin: 0 auto;" src="images/哨岗场地分区.jpg" width="80%" alt="" /></p>   
<p align="center">哨岗场地分区</p>  
<p align="center"><img style="display: block; margin: 0 auto;" src="images/哨岗识别计算坐标.gif" width="80%" alt="" /></p>   
<p align="center">哨岗识别计算坐标</p>  
<p align="center"><img style="display: block; margin: 0 auto;" src="images/机器人及其装甲板识别.gif" width="80%" alt="" /></p>   
<p align="center">机器人及其装甲板识别</p>
<p align="center"><img style="display: block; margin: 0 auto;" src="images/tensorRT加持的yolov4-tiny测试.gif" width="80%" alt="" /></p>   
<p align="center">tensorRT加持的yolov4-tiny测试</p>  
<p align="center"><img style="display: block; margin: 0 auto;" src="images/5.6米识别机器人装甲板.gif" width="80%" alt="" /></p>   
<p align="center">5.6米识别机器人装甲板model_size(448, 256) FPS40</p>   
<p align="center"><img style="display: block; margin: 0 auto;" src="images/7.8米识别机器人装甲板.gif" width="80%" alt="" /></p>   
<p align="center">7.8米识别机器人装甲板model_size(512, 418) FPS30</p>   

## **3. 依赖工具，软、硬件环境**
####**软件部分：**   

系统版本：Ubuntu18.04    

机载平台(jetson):   
CUDA 10.2    
python3.6   
PyTorch1.6    
OpenCV3.4.x   
Jetpack 4.4   
PyYAML>=5.3     
scipy==1.4.1  
tqdm>=4.41.0    
numpy>=1.18.5   
TensorFlow==2.2   
matplotlib>=3.2.2     
torchvision>=0.7.0      
OpenCV-python>=4.1.2    
tensorRT: jetpack 4.4刷机时安装即可  

哨岗电脑(mi pro)：   
CUDA 10.1     
python3.6   
PyTorch1.6    
OpenCV3.4.x   
PyYAML>=5.3     
scipy>=1.4.1  
tqdm>=4.41.0    
numpy>=1.18.5   
matplotlib>=3.2.2     
torchvision>=0.7.0      
OpenCV-python>=4.1.2 


####**硬件部分：**   
机载平台： Jetson AGX Xavier  
哨岗电脑： 小米Pro  Intel i7-8550U @1.8G + GeForce MX150      
单目摄像头：威鑫视界WX605摄像头，镜头150°，焦距2.45mm，分辨率1280*720，帧率 120   
深度相机： Intel D435i深度相机，RGB：1920x1080, 30FPS, 深度图像：最高1280x720主动立体深度图，FPS: 90 max  


# **4. 编译、安装方式**

## **对于哨岗的测试**：  
### **linux下**：  
① 克隆代码至本地仓库：git clone https://github.com/LHL6666/perception.git Wust_Perception   
② 添加该工程下Innocent_Bird-master项目路径到python的搜索路径，使python解释器能够找到models,runs,utils文件夹下的python文件，或者直接将这三个文件夹复制到    
python site-packages里面，例如~/.local/lib/python3.6/site-packages/    
③ 修改Innocent_Bird.py文件，更改模型搜索路径，更改摄像头的编号后，打开终端cd到Innocent_Bird-master工程的根目录下面，直接运行 python3 Innocent_Bird.py     
### **window10下**：  
① 下载工程：     
② 使用pycharm等软件打开Innocent_Bird-master工程文件    
③ 修改Innocent_Bird.py里面的weights路径和VideoCapture相机编号，0为电脑自带摄像头   
③ 直接运行Innocent_Bird.py      

## **对于机载视觉功能包测试**： 
### **linux下**：  
① 将下载好的LHL_RoboRTS工作空间放到home下  
② 打开终端，切换路径到~/LHL_RoboRTS/下  
③ catkin_make, 添加"source ~/LHL_RoboRTS/devel/setup.bash"到.bashrc文件中  
④ 进入到~/LHL_RoboRTS/src/my_roborts_camera/bin/下，给这里的每个文件添加可执行文件权限  
⑤ 可选 打开~/LHL_RoboRTS/src/my_roborts_camera/src/Python_package/image_capture.py文件，修改采用的摄像头编号，一般机载PC不用改  
⑥ 打开~/LHL_RoboRTS/src/my_roborts_camera/src/Python_package/LHL_Car_Str_Detection.py文件，修改模型绝对路径，保存之后直接退出即可  
⑦ 启动rosmaster, 运行image_capture、image_after、LHL_Car_Str_Detection和car_armor_position_subscriber分别观察窗口的输出情况和位置信息等  
  指令为：rosrun my_roborts_camera + 以上可执行文件(eg: image_capture)

# **5. 软件使用说明** 
## ***A. 深度学习目标检测算法***   


## 数据集  
yolov4-tiny使用的是voc格式的标签，ultralytics yolov5使用的是yolo格式的标签，不过在该工程中提供了voc转yolo格式的Convert_xml_to_txt.py文件。
① 哨岗搭载的模型训练用的数据集一共250张左右，其中验证数据集50左右，在小米笔记本pro上(MX150入门显卡)200epochs, batch_size 16, train_size和test_size为256时训练时间仅仅为0.65个小时，mAP@0.5接近1，可在下面链接下载数据集  
② 机器人搭载的模型训练用的数据集一共1000张左右，其中包含了验证数据集200张左右，在小米笔记本pro上300 epochs, batch_size 8, train_size和test_size为480时训练时间6.8个小时左右，在jetson agx xavier上 300 epochs, batch_size 128, train_size和test_size为480时训练时间仅仅为2.7个小时左右， 由于该数据集比较大，不好上传暂不开源。（实际结果可能会有偏差，非严格测试）  


## 模型  

模型大小为14MB左右，已经很小  

### 哨岗检测模型
链接：https://pan.baidu.com/s/15dIvgZN781N9q14gnpFfZw      
提取码：wifw       

### 机载检测模型
链接：https://pan.baidu.com/s/1HHIdqMT0tnO45W5gij9R0A      
提取码：5twr    

### yolov5s权重
链接: https://pan.baidu.com/s/1Ge1--weNoh_KgB2xFRRQ2A  	
提取码: m0hl

### 哨岗训练数据集
链接：https://pan.baidu.com/s/1fuAy0An9HTO2rey9KgZsZQ 
提取码：oufy

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
│   ├── datasets.py
│   ├── googles_utils.py
│   ├── torch_utils.py
│   └── utils.py
│ 
├── Camera_Calibration.py
├── Convert_xml_to_txt.py
├── Innocent_Bird.py // 
├── test.py // 
├── train.py // 
└── requirements.txt // 
```  

# **7. 原理介绍与理论支持分析**   
## 1. 机器人与装甲板识别及哨岗识别  


## 2. 机器人姿态估计  
  

### 3. 机器人运动预测
初步测试了KCF、MOSSE和CSRT等传统跟踪算法，发现MOSSE算法对该哨岗视觉算法最合适的，在机器人被遮挡大部分时仍能够正常跟踪不容易丢失目标，
KCF虽然能够达到300多帧的跟踪速度，但是精度和抗干扰性都不是很好，MOSSE在我的测试过程中保持了120帧左右的跟踪速度，精度和抗干扰性好很
多。但是由于第一届参加比赛还没有得到固定场地，还没录视频就被迫更换场地，没法固定哨岗相机不满足测试条件了

# **8. 数据流图及软件框图**  
<p align="center"><img style="display: block; margin: 0 auto;" src="My_PIC/软件框图.png" width="30%" alt="" /></p>  
<p align="center">图8-1 软件框图</p>  
<p align="center"><img style="display: block; margin: 0 auto;" src="My_PIC/总框图.png" width="80%" alt="" /></p>  
<p align="center">图8-2 总数据流框图</p>  


### Reference   
https://docs.opencv.org/master/d9/df8/tutorial_root.html    
https://github.com/ultralytics/yolov5   
