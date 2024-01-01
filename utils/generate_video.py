import cv2
import numpy as np
import glob
import os


def generate_video_from_frames(output_video_path, fps):
    # Get the filenames of all images in the folder
    width = 0
    height = 0
    frames = []
    # Get paths of frames, ensuring they are sorted in order
    frame_paths = sorted(glob.glob("./marked_images/*.jpg"), key=lambda x: int(os.path.basename(x).split('.')[0]))
    for frame_path in frame_paths:
        frames.append(cv2.imread(frame_path))
    #  Read the first image to obtain image dimensions
    if frames is not None:
        height = frames[0].shape[0]
        width = frames[0].shape[1]
        print(height)
    else:
        print("Failed to load the image.")

    # create video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Write video frame by frame
    for frame_path in frame_paths:
        frame = cv2.imread(frame_path)
        out.write(frame)

    # release video
    out.release()

    print("Video generation complete")
