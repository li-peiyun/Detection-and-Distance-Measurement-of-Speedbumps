import cv2
import os
from utils.file_operations import cleanFolder

# Used to take a picture of the panel to calibrate the camera
def Capture(folder: str, num: int, cap: cv2.VideoCapture, isPreClean=False) -> None:
    if isPreClean == True:
        cleanFolder(folder)

    count = 0   # nums of photos captured this time
    images_count = len(os.listdir(folder))  # num of current images
    id = images_count  # don't +1 because id starts from 0

    while True:
        success, frame = cap.read()
        if (num != -1) and (not success or count >= num):
            break

        cv2.imshow('press c to capture , q to exit..', frame)

        key = cv2.waitKey(1) & 0xff
        if key == ord('q') or key == ord('Q'):
            break

        elif key == ord('c'):
            img_path = f'{folder}/{id}.png'  # save into the folder
            cv2.imwrite(img_path, frame)
            print(f'captured at {img_path}')
            id += 1