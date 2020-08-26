## 该项目是ICRA2020 AI Challenge ChongShi战队的感知功能代码说明 --version1.0.1
### 该仓库包含了哨岗视觉Innocent_Bird-master工程和运行于机器人上面的视觉功能包my_roborts_camera

## copyright@LHL of Wust ChongShi team
## **1. 软件功能介绍**  
哨岗感知功能实现了敌我机器人识别、敌方机器人装甲板识别、敌方机器人位置坐标识别；机器人感知部分实现了在ROS工作空间里通过自定义图像类型进行发布和订阅解码，进行图像检测并发布红蓝方及其装甲板置信度最高的机器人和装甲板位置坐标，也同样实现了敌我机器人识别，装甲板类型识别，姿态识别，障碍物编号识别等功能。
#### **机器人和装甲板识别算法框架：**    
使用了深度学习目标检测算法：一共尝试了**yolov4, yolov4-tiny, "ultralytics yolov5", AlexeyAB发行的tensorRT加速版yolov4-tiny**框架，其中tensorRT加持的yolov4-tiny算法在我们的机载PC（仅为ARM处理器）上识别速度能够高达近乎150帧，能够在三米内稳定识别装甲板和尾灯，7米之内稳定识别机器人（有一定的防遮挡能力），但是由于对装甲板的识别距离太近，限制了日后决策组发展，因此最后采取折中的方案，最终在哨岗视觉和机器人视觉处理上使用了来自ultralytics公司的"yolov5"框架，该框架在调节模型大小和优化器等参数后取得较好的效果，能够在6米内稳定识别装甲板和尾灯，识别机器人的准确度也很高。  
使用机载PC测试结果如下（以最后模型为准）：   

|                |ultralytics yolov5 |AlexeyAB版 yolov4-tiny                        |
|----------------|-------------------------------|-----------------------------|
|6米距离识别机器人  |  94%   |  80%  |
|6米距离识别装甲板  |  72%   |  NO   |
|稳定识别时的FPS(size不同)  |30-85   |   50-150|


## **2. 软件效果展示**   
#### **若无法加载图像，建议下载工程后到Innocent_Bird-master/images/文件夹下打开** 
<p align="center"><img style="display: block; margin: 0 auto;" src="https://github.com/LHL6666/perception/blob/master/Wust_perception/Innocent_Bird-master/images/整车.jpg" width="80%" alt="" /></p>   
<p align="center">整车结构图</p>   
<p align="center"><img style="display: block; margin: 0 auto;" src="https://github.com/LHL6666/perception/blob/master/Wust_perception/Innocent_Bird-master/images/场地字符识别.jpg" width="80%" alt="" /></p>   
<p align="center">场地字符识别</p>  
<p align="center"><img style="display: block; margin: 0 auto;" src="https://github.com/LHL6666/perception/blob/master/Wust_perception/Innocent_Bird-master/images/哨岗识别计算坐标.gif" width="80%" alt="" /></p>   
<p align="center">哨岗识别计算坐标</p>   
<p align="center"><img style="display: block; margin: 0 auto;" src="https://github.com/LHL6666/perception/blob/master/Wust_perception/Innocent_Bird-master/images/哨岗场地分区图.jpg" width="80%" alt="" /></p>   
<p align="center">哨岗场地分区图</p>
<p align="center"><img style="display: block; margin: 0 auto;" src="https://github.com/LHL6666/perception/blob/master/Wust_perception/Innocent_Bird-master/images/tensorRT加持的yolov4-tiny测试.gif" width="80%" alt="" /></p>   
<p align="center">tensorRT加持的yolov4-tiny测试</p>  
<p align="center"><img style="display: block; margin: 0 auto;" src="https://github.com/LHL6666/perception/blob/master/Wust_perception/Innocent_Bird-master/images/镜头划伤起雾时机器人及其装甲板识别.gif" width="80%" alt="" /></p>   
<p align="center">镜头划伤起雾时机器人及其装甲板识别</p> 
<p align="center"><img style="display: block; margin: 0 auto;" src="https://github.com/LHL6666/perception/blob/master/Wust_perception/Innocent_Bird-master/images/5.6米识别机器人装甲板.gif" width="80%" alt="" /></p>   
<p align="center">5.6米识别机器人装甲板model_size(448, 256) FPS40左右</p>   
<p align="center"><img style="display: block; margin: 0 auto;" src="https://github.com/LHL6666/perception/blob/master/Wust_perception/Innocent_Bird-master/images/7.8米识别机器人装甲板.gif" width="80%" alt="" /></p>   
<p align="center">7.8米识别机器人装甲板model_size(512, 418) FPS30左右</p>   
<p align="center"><img style="display: block; margin: 0 auto;" src="https://github.com/LHL6666/perception/blob/master/Wust_perception/Innocent_Bird-master/images/Ros中机器人感知测试.gif" width="80%" alt="" /></p>   
<p align="center">Ros中机器人机载PC感知测试(512, 414) FPS20左右(录屏后机载电脑cpu100%)</p> 


## **3. 依赖工具，软、硬件环境**
#### **软件部分：**   

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

数据集标注软件：labelimg  

#### **硬件部分：**   
机载平台： Jetson AGX Xavier  
哨岗电脑： 小米Pro  Intel i7-8550U @1.8G 8G + GeForce MX150      
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

# **5. 文件目录结构及文件用途说明**   
 ```
Innocent_Bird-master.
├── models
│   ├── common.py // 包含了yolov5s、yolov5m、yolov5l和yolov5x模型通用的模块，还包括了SPP等结构
│   ├── export.py // 将训练好的.pt模型转换成onnx和TorchScript格式，减少由于训练模型时保存的设备和时间参数等，增加通用性以及可用于tensorRT加速处理
│   ├── experimental.py // 包含实验模块还有加载训练好的模型函数，比较新颖的MixConv2d混合神经网络模块都在里面有体现
│   └── yolo.py  //模型文件，包含了用来解析输入的yolov5s.yaml参数网络的功能
│
├── utils
│   ├── activations.py //激活函数文件，里面分成Swish激活函数实现和Mish激活函数实现
│   ├── datasets.py // 数据集文件，包括训练测试时加载数据集进行处理以及加载(多)图片和(多)视频流用于识别检测
│   ├── googles_utils.py // 这个文件包含了google utils功能，在没有发现本地模型的时候自动到google下载，下载谷歌驱动等功能
│   ├── torch_utils.py // torch的utils功能，如time_synchronized获取cuda可用时同步时间，选择gpu/cpu设备，绘制逐行描述一个PyTorch模型等功能
│   └── utils.py // 相当于脚本文件，包含了检查文件是否存在，计算平均精度，检查图像size与模型是否匹配等功能
│ 
├── Camera_Calibration.py // 矫正畸变后的图像，用来收集数据集使用
├── Convert_xml_to_txt.py // voc数据集转yolo数据集
├── Innocent_Bird.py // 哨岗检测文件
├── test.py // 大部分与train.py功能相同，该部分主要用于运行train时，计算每个epoch的mAP。
├── train.py // 训练用的文件
└── requirements.txt // 环境依赖说明
```


```
.
├── LHL_RoboRTS
│   ├── src
│   │   ├── my_roborts_camera // 视觉功能包
|   │   │   ├── bin // 存放可执行文件的文件夹
|   |   │   │   ├── car_armor_position_subscriber // 机器人、装甲板以及位置信息的car_armor_position_subscriber.py订阅文件对应的可执行文件
|   |   │   │   ├── image_after // 图像传输中介image_after.py对应的可执行文件
|   |   │   │   ├── image_capture // 读取摄像头视频流image_capture.py文件对应的可执行文件
|   |   │   │   ├── LHL_Car_Str_Detection // 进行检测跟踪机器人装甲板等类的可执行文件，对应于LHL_Car_Str_Detection.py
│   │   │   ├── msg // 消息文件夹
|   |   │   │   ├── my_msg.msg // 自定义的图像类型消息，用于解决python3无法直接使用CV_bridge的问题（在image_after.py和LHL_Car_Str_Detection.py中体现）
|   |   │   │   ├── car_armor_position.msg // 机器人和装甲板还有临时目标的位置信息存放文件
│   │   │   ├── src // 源码文件
|   |   │   │   ├── Python_package // 存放python文件的文件夹
|   |   |   │   │   ├── __pycache__
|   |   |   |   │   ├── car_armor_position_subscriber.py // 机器人、装甲板以及位置信息的信息订阅实现文件
|   |   |   |   │   ├── car_armor_position_subscriber.pyc //
|   |   |   |   │   ├── image_after.py // 图像传输中介的文件
|   |   |   |   │   ├── image_after.pyc //
|   |   |   |   │   ├── image_capture.py // 读取摄像头图像并发布的文件
|   |   |   |   │   ├── image_capture.pyc //
|   |   |   |   │   ├── LHL_Car_Str_Detection.py // 进行检测跟踪机器人、装甲板、尾灯和障碍物字符编号的文件
│   │   │   ├── CMakeLists.txt // 编译配置文件，添加依赖项等
│   │   │   ├── package.xml // 描述文件
│   │   │   ├── setup.py
│   │   ├── CMakeLists.txt
```

# **6. 软件使用说明** 
## ***A. 深度学习目标检测算法框架修改***   
修改的网络: ultralytics团队的"yolov5"框架   
  yolov5s的网络结构和yolov4是基本相同的，网络结构每一层的输入都是上一层的输出，所以为了方便使用者修改网络结构，就提取出了depth_multiple和width_multiple两个参数，这两个参数和网络层结构配置被放在models/yolov5s.yaml文件中，只需要修改depth_multiple和width_multiple参数即可修改网络模型的结构。其中depth_multiple深度神经因子参数调节的是非功能层(对conv,spp,Focus等层不起作用)，如瓶颈层BottleneckCSP，它控制的是神经网络的深度，width_multiple参数修改了卷积层数，width_multiple=0.5即指卷积层数目减少到原来默认值的一半。对于骨干网络backbone，下采样使得特征图从大到小，深度逐渐加深，对于头部结构head，可以看到头部层升维再降维变化，ultralytics团队更新代码后采用了PN Net结构。  
  分析推测：   
① 增加一层先验框anchors为[5,6, 7,9, 12,10]，应该能够更准确地检测小物体    
② 调整yolov5s.yaml参数number的个数应该能调出更好的模型，甚至使得BottleneckCSP层中再包含多个BottleneckCSP层，但是模型可能更大推理速度变慢    
③ 在上采样瓶颈层处理后尝试增加SELayer层对上一层的特征图深度进行加权处理，对于检测类型的任务应该能够加快收敛，训练速度和检测精度都应该有所提升   
  最后，我尝试了增加anchors和SELayer层，anchors在修改之后在我们场地测试时发现对远处的装甲板确实能够标记的更准确了，但是误判率却又有所增加，所以yolov5s.yaml中我将新增的anchors注释了，后面才了解到yolov5中先验框的大小会在训练过程自动调节，已经适配地相当优秀了。SELayer的实现原理是先做平均赤化和线性分类，然后使用relu激活函数约束后再次线性分类，最后加上Sigmoid处理。  

####修改前网络结构:  
```
                 from  n    params  module                                  arguments                     
  0                -1  1      3520  models.common.Focus                     [3, 32, 3]                    
  1                -1  1     18560  models.common.Conv                      [32, 64, 3, 2]                
  2                -1  1     19904  models.common.BottleneckCSP             [64, 64, 1]                   
  3                -1  1     73984  models.common.Conv                      [64, 128, 3, 2]               
  4                -1  1    161152  models.common.BottleneckCSP             [128, 128, 3]                 
  5                -1  1    295424  models.common.Conv                      [128, 256, 3, 2]              
  6                -1  1    641792  models.common.BottleneckCSP             [256, 256, 3]                 
  7                -1  1   1180672  models.common.Conv                      [256, 512, 3, 2]              
  8                -1  1    656896  models.common.SPP                       [512, 512, [5, 9, 13]]        
  9                -1  1   1248768  models.common.BottleneckCSP             [512, 512, 1, False]          
 10                -1  1    131584  models.common.Conv                      [512, 256, 1, 1]              
 11                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']          
 12           [-1, 6]  1         0  models.common.Concat                    [1]                           
 13                -1  1    378624  models.common.BottleneckCSP             [512, 256, 1, False]          
 14                -1  1     33024  models.common.Conv                      [256, 128, 1, 1]              
 15                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']          
 16           [-1, 4]  1         0  models.common.Concat                    [1]                           
 17                -1  1     95104  models.common.BottleneckCSP             [256, 128, 1, False]          
 18                -1  1    147712  models.common.Conv                      [128, 128, 3, 2]              
 19          [-1, 14]  1         0  models.common.Concat                    [1]                           
 20                -1  1    313088  models.common.BottleneckCSP             [256, 256, 1, False]          
 21                -1  1    590336  models.common.Conv                      [256, 256, 3, 2]              
 22          [-1, 10]  1         0  models.common.Concat                    [1]                           
 23                -1  1   1248768  models.common.BottleneckCSP             [512, 512, 1, False]          
 24      [17, 20, 23]  1     37758  Detect                                  [9, [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]], [128, 256, 512]]
Model Summary: 191 layers, 7.27667e+06 parameters, 7.27667e+06 gradients
```

####增加SELayer加权处理后的网络结构:  
```                 from  n    params  module                                  arguments                     
  0                -1  1      3520  models.common.Focus                     [3, 32, 3]                    
  1                -1  1     18560  models.common.Conv                      [32, 64, 3, 2]                
  2                -1  1     19904  models.common.BottleneckCSP             [64, 64, 1]                   
  3                -1  1     73984  models.common.Conv                      [64, 128, 3, 2]               
  4                -1  1    161152  models.common.BottleneckCSP             [128, 128, 3]                 
  5                -1  1      2048  models.common.SELayer                   [128, 16]                     
  6                -1  1    295424  models.common.Conv                      [128, 256, 3, 2]              
  7                -1  1    641792  models.common.BottleneckCSP             [256, 256, 3]                 
  8                -1  1      8192  models.common.SELayer                   [256, 16]                     
  9                -1  1   1180672  models.common.Conv                      [256, 512, 3, 2]              
 10                -1  1    656896  models.common.SPP                       [512, 512, [5, 9, 13]]        
 11                -1  1     32768  models.common.SELayer                   [512, 16]                     
 12                -1  1   1248768  models.common.BottleneckCSP             [512, 512, 1, False]          
 13                -1  1    131584  models.common.Conv                      [512, 256, 1, 1]              
 14                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']          
 15           [-1, 6]  1         0  models.common.Concat                    [1]                           
 16                -1  1    378624  models.common.BottleneckCSP             [512, 256, 1, False]          
 17                -1  1     33024  models.common.Conv                      [256, 128, 1, 1]              
 18                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']          
 19           [-1, 4]  1         0  models.common.Concat                    [1]                           
 20                -1  1     95104  models.common.BottleneckCSP             [256, 128, 1, False]          
 21                -1  1    147712  models.common.Conv                      [128, 128, 3, 2]              
 22          [-1, 14]  1         0  models.common.Concat                    [1]                           
 23                -1  1    345856  models.common.BottleneckCSP             [384, 256, 1, False]          
 24                -1  1    590336  models.common.Conv                      [256, 256, 3, 2]              
 25          [-1, 10]  1         0  models.common.Concat                    [1]                           
 26                -1  1   1379840  models.common.BottleneckCSP             [768, 512, 1, False]          
 27      [17, 20, 23]  1     21630  Detect                                  [9, [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]], [128, 128, 256]]
Reversing anchor order
Model Summary: 197 layers, 7.46739e+06 parameters, 7.46739e+06 gradients
```

## 数据集  
由于拍摄的数据集前后相关性不大，因此未采用视频标注工具而使用了labelimg的标注软件  
yolov4-tiny使用的是voc格式的标签，ultralytics yolov5使用的是yolo格式的标签，不过在该工程中提供了voc转yolo格式的Convert_xml_to_txt.py文件。
① 哨岗搭载的模型训练用的数据集一共250张左右，其中验证数据集50左右，在小米笔记本pro上(MX150入门显卡)200epochs, batch_size 16, train_size和test_size为256时训练时间仅仅为0.65个小时，mAP@0.5接近1，可在下面链接下载数据集  
② 机器人搭载的模型训练用的数据集一共1000张左右，其中包含了验证数据集200张左右，在小米笔记本pro上300 epochs, batch_size 8, train_size和test_size为480时训练时间6个小时左右，在jetson agx xavier上 300 epochs, batch_size 128, train_size和test_size为480时训练时间仅仅为2个小时左右， 由于该数据集比较大，不好上传暂不开源。（实际结果可能会有偏差，非严格测试）  
③ 训练数据集文件结构：  
```
.
├── DataSet_V5
│   ├── test // 测试数据集
│   │   ├── images
│   │   ├── labels
│   ├── train // 
│   │   ├── images // 训练数据集的图片
│   │   ├── labels // 训练数据集的标签
│   ├── valid // 
│   │   ├── images // 验证数据集的图片
│   │   ├── labels // 验证数据集的标签
│   ├── data.yaml // classes的总数以及名称，训练测试数据集的路径配置
```

## 模型  

模型大小仅仅14MB左右 

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

  

# **7. 原理介绍与理论支持分析**   
## 1. 哨岗识别原理与流程  
① 摄像头矫正得到相机参数用于OpenCV remap，得到无畸变图像   
② 使用逆透视算法对梯形畸变进行矫正，得到了只有半个场地区域大小的俯视图    
③ 增加保存图像功能，收集数据集并标准数据集    
④ 改进ultralytics公司开源的yolov5框架来训练红蓝车和装甲板模型    
⑤ 使用训练好的模型对半场地图像进行检测识别，两个哨岗摄像头分别负责一半场地，互相独立       
⑥ 根据比赛场地的长宽信息，鸟瞰图中机器人的相对坐标，由比例关系可以计算得到实际的坐标信息     
⑦ 将识别到的敌方机器人位置及其装甲板位置信息（置信度最高的）发布到innocent_msg消息中，（由于只有一台机器人，暂时未在移动PC上实现测试）       

#### 坐标的简单计算如下所示  

#### 哨岗视角建立坐标系图
<p align="center"><img style="display: block; margin: 0 auto;" src="https://github.com/LHL6666/perception/blob/master/Wust_perception/Innocent_Bird-master/images/哨岗场地分区图.jpg" width="80%" alt="" /></p>   
<p align="center">哨岗场地分区</p>  

```
# 场地半宽x=254cm, y0=340cm, y1=354cm  ，得出的car_x, car_y为该种坐标系下实际的1:1坐标
# adjust_r为调整系数，field_y1为y1(逆透视图中图像底部到参考点对应的实际场地垂直距离), field_y0即指y0(逆透视图中由参考点到图像顶部对应的实际场地垂直距离), 具体见上图  
adjust_r = field_y1 / field_y0  
# Car_Center[0]指逆透视图中机器人在height方向上的位置，Car_Center[0]指逆透视图中机器人在width方向上的位置  
car_y = ((ref_point[1] - Car_Center[1]) / ref_point[1]) * field_y0 * adjust_r  
car_x = ((Car_Center[0] - ref_point[0]) / Bird_img.shape[1]) * field_x * adjust_r  
```

## 2. 机器人姿态估计  
  由于武汉批准返校时间太短太短，加上第一次参赛经验不足，因此姿态检测方面只靠识别机器人尾灯和装甲板的分布来推测姿态信息，AI机器人的防护做得比较好，麦轮已经被遮挡了一半，仅仅根据麦轮来解算得出姿态信息可信度大大降低，而且每台AI机器人上面的器件摆放位置以及样式多少都会有差异，机器人全黑的配色让我们不能简单通过深度学习来识别区分大部分机器人姿态。因此针对AI机器人姿态检测的困难性
  

### 3. 机器人运动预测
初步测试了KCF、MOSSE和CSRT等传统跟踪算法，发现MOSSE算法(Minimum Output Sum of SquaredError)对该视觉检测算法最合适的，在机器人被遮挡大部分时仍能够正常跟踪不容易丢失目标，KCF虽然能够达到300多帧的跟踪速度，但是精度和抗干扰性都不是很好，MOSSE在我的测试过程中保持了120帧左右的跟踪速度，精度和抗干扰性好很多。但是由于第一届参加比赛还没有得到固定场地，还没录视频就被迫更换场地，没法固定哨岗相机不满足测试条件了

# **8. 数据流图及软件框图**  
##### 若无法加载图像，建议下载工程后到Innocent_Bird-master/images/文件夹下打开
<p align="center"><img style="display: block; margin: 0 auto;" src="https://github.com/LHL6666/perception/blob/master/Wust_perception/Innocent_Bird-master/Data_diagram_image/哨岗流程图.jpg" width="30%" alt="" /></p>  
<p align="center">图8-1 哨岗流程图</p>  
<p align="center"><img style="display: block; margin: 0 auto;" src="https://github.com/LHL6666/perception/blob/master/Wust_perception/Innocent_Bird-master/Data_diagram_image/机载模型参数评估图.png" width="80%" alt="" /></p>  
<p align="center">图8-2 机载模型参数评估图</p>  
<p align="center"><img style="display: block; margin: 0 auto;" src="https://github.com/LHL6666/perception/blob/master/Wust_perception/Innocent_Bird-master/Data_diagram_image/修改过的网络框架.jpg" width="30%" alt="" /></p>  
<p align="center">图8-1 修改过的网络框架</p>  
<p align="center"><img style="display: block; margin: 0 auto;" src="https://github.com/LHL6666/perception/blob/master/Wust_perception/Innocent_Bird-master/Data_diagram_image/AI_硬件框图.jpg" width="80%" alt="" /></p>  
<p align="center">图8-4 AI_硬件框图</p>  
<p align="center"><img style="display: block; margin: 0 auto;" src="https://github.com/LHL6666/perception/blob/master/Wust_perception/Innocent_Bird-master/Data_diagram_image/使用jetson%20agx%20xavier训练模型时长.png
" width="80%" alt="" /></p>  
<p align="center">图8-3 使用jetson agx xavier训练模型时长</p>  



# **9. 解决的工程问题和创新之处**   
- [x] 解决了jetson agx xavier安装最新深度学习环境jetpack4.4和高版本下运行官方RoboRTS ROS工作空间无法显示地图和节点发布不全的问题    
- [x] 解决了python3环境下无法直接使用CV_bridge的问题，不需要建立虚拟环境和单独编译python3专用的CV_bridge（在image_after.py和LHL_Car_Str_Detection.py中体现）   
- [x] 对数据集中出现的未显示机器人编号但是能看到颜色特征的机器人进行了特定处理(例如机器人编号被遮挡有红色特征都归为red_car2)，减少了识别classes数目，更及时地反馈敌方机器人信息。    
- [x] 解决了哨岗视觉机器人定位不准的问题，定位精确度高达90%以上    
- [x] 参考yolo检测代码，编写了自己的detection文件(Innocent_Bird.py, LHL_Car_Str_Detection.py)，代码已经尽量简化明了，运行速度较原代码有所提高，能够用于ros工作空间下面运行不依靠封装良好的Darknet结构，并且对红蓝车和装甲板尾灯都指定了特定的可视化标记，例如红方机器人方框颜色为红色，装甲板2号为天蓝色，置信度低时为灰色等(哨岗和机载检测有差异)   

### Reference   
https://github.com/AlexeyAB/darknet   
https://github.com/ultralytics/yolov5   
https://developer.nvidia.com/embedded/jetpack   
https://docs.opencv.org/master/d9/df8/tutorial_root.html      
 

