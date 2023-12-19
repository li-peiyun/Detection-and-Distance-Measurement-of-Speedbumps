# 减速带检测和测距

## 编译环境

#### 创建编译环境

这里使用 Anaconda 创建用于运行 YOLOv8 的虚拟环境，如果没有 Anaconda 可以手动添加以下依赖项或使用python终端下载。

使用命令行，创建用于运行 YOLOv8 的虚拟环境并激活

```
conda create -n yolov8 python=3.8

conda activate yolov8
```

在激活的 yolov8 环境中，通过cd命令跳转到该项目 Detection-and-Distance-Measurement-of-Speedbumps 文件夹地址。

执行以下命令安装所需依赖项：

```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu11834
pip install -r requirements.txt 
pip install ultralytics
pip install -e.
```

#### 在 pycharm 中使用上述 conda 虚拟环境

【设置】-【项目】-【Python 解释器】-【添加】-【Conda 环境】-【现有环境】：

解释器：上述创建的 yolov8 环境的地址，可能为 `(自己的下载地址)\anaconda3\envs\yolov8\python.exe`

Conda 可执行文件：`(自己的下载地址)\anaconda3\Scripts\conda.exe`

## 准备工作

#### 一、相机标记（乐怡）

计算相机参数，去畸变

设备：鱼眼相机、棋盘格

#### 二、单应矩阵标定（李欣）

去畸变 -> 地面坐标系

设备：鱼眼相机、小车、地面关键点（瓷砖）

#### 三、减速带检测模型训练（李佩耘）

减速带检测

工具：yolov8（使用老师提供的数据集进行训练）

## 实现

#### 一、图像去畸变（乐怡）

具体要求见detect.py

文件夹含义：capturedVideo是用来保存直接拍摄到的视频，

video_and_undistortedVideo保存了我们拍摄的视频及其对应的去畸变后的视频，

video中保存了一个拆分与去畸变的测试视频，

generated_video是最后由照片合成的视频，

undistorted_image保存了原始视频被拆分后的去畸变的帧图片，

UndistortionImg_of_chessboard是测试用的标定板去畸变图片，

Img_for_calibration是标定用的标定板图片，

TileImage是拍摄的瓷砖图片。

#### 二、减速带检测（李佩耘）

已完成，见detect.py

#### 三、减速带距离计算（李欣）

具体要求见detect.py