import cv2
import os
from configs.camera_intrinsic import cameraMatrix, distCoeff


def split_video_to_undistroted_frames(videopath):
    # Camera parameters
    camera_matrix = cameraMatrix
    dist_coeff = distCoeff
    # Open the video file
    cap = cv2.VideoCapture(videopath)

    # Check if the video is successfully opened
    if not cap.isOpened():
        print("Error: 无法打开视频文件")
        return

    # Get the frame rate and total number of frames in the video
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"帧速率: {fps}, 总帧数: {frame_count}")

    # Read the first frame of the video
    success, oriframe = cap.read()

    # Create a folder to save the split images
    out_undistorted_folder = 'undistorted_image'
    os.makedirs(out_undistorted_folder, exist_ok=True)

    frame_count = 0

    while success:
        # Undistort the video frame
        undistorted_frame = cv2.fisheye.undistortImage(oriframe, camera_matrix, dist_coeff, None, camera_matrix)
        # Save the current frame to the output folder
        # Split and save each frame as an image
        frame_filename = os.path.join(out_undistorted_folder, f"frame_{frame_count:04d}.png")
        cv2.imwrite(frame_filename, undistorted_frame)

        # Read the next frame
        success, oriframe = cap.read()

        frame_count += 1

    # Release the video object
    cap.release()

    print("Splitting completed")

