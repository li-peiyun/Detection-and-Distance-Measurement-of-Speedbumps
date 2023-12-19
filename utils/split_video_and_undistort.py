import cv2
import os
from configs.camera_intrinsic import cameraMatrix, distCoeff


def split_video_to_undistroted_frames(videopath):
    camera_matrix = cameraMatrix
    dist_coeff = distCoeff
    # 打开视频文件
    cap = cv2.VideoCapture(videopath)

    # 检查视频是否成功打开
    if not cap.isOpened():
        print("Error: 无法打开视频文件")
        return

    # 获取视频的帧速率和帧数
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"帧速率: {fps}, 总帧数: {frame_count}")

    # 逐帧读取视频
    success, oriframe = cap.read()

    # 创建文件夹用于保存拆分的图片
    out_undistorted_folder = 'undistorted_image'
    os.makedirs(out_undistorted_folder, exist_ok=True)

    frame_count = 0

    while success:
        # 对视频帧进行去畸变
        undistorted_frame = cv2.fisheye.undistortImage(oriframe, camera_matrix, dist_coeff, None, camera_matrix)
        # 保存当前帧到输出文件夹
        # 拆分并保存每一帧为图片
        frame_filename = os.path.join(out_undistorted_folder, f"frame_{frame_count:04d}.png")
        cv2.imwrite(frame_filename, undistorted_frame)

        # 读取下一帧
        success, oriframe = cap.read()

        frame_count += 1

    # 释放视频对象
    cap.release()

    print("拆分完成")

