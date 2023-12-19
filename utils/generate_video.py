import cv2
import numpy as np
import glob
import os

def generate_video_from_frames(output_video_path, fps):
    # 获取文件夹中所有图片的文件名
    width = 0
    height = 0
    frames = []
    frame_paths = glob.glob("./undistorted_image/frame_****.png")
    frame_paths.sort()  # 确保图片按照顺序排列
    for frame_path in frame_paths:
        frames.append(cv2.imread(frame_path))
    # 读取第一张图片以获取图像大小
    if frames is not None:
        height = frames[0].shape[0]
        width = frames[0].shape[1]
        print(height)
    else:
        print("Failed to load the image.")

    # 创建视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # 逐帧写入视频
    for frame_path in frame_paths:
        frame = cv2.imread(frame_path)
        out.write(frame)

    # 释放视频写入对象
    out.release()

    print("视频生成完成")
