import cv2
import os
from configs.camera_intrinsic import cameraMatrix, distCoeff

# This file is used for shooting video, as well as shooting video while generating the dedistorted video

def CaptureVideo(output_folder):
    inputVideo = cv2.VideoCapture(1)
    if not inputVideo.isOpened():
        print("Error: Could not open camera at index 1")

    # Create the output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = inputVideo.get(cv2.CAP_PROP_FPS)
    width, height = int(inputVideo.get(cv2.CAP_PROP_FRAME_WIDTH)), int(inputVideo.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path = os.path.join(output_folder, 'captured_video.avi')
    outVideo = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = inputVideo.read()
        if ret:
            # Display the frame
            cv2.imshow('Capturing Video', frame)
            outVideo.write(frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("Error: Failed to capture video.")
            break

    # Release resources
    inputVideo.release()
    outVideo.release()
    cv2.destroyAllWindows()

def undistort_video(output_folder):
    # Camera parameter
    camera_matrix = cameraMatrix
    dist_coeff = distCoeff
    # open camera
    inputVideo = cv2.VideoCapture(1)

    os.makedirs(output_folder, exist_ok=True)

    # Set up video saving
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_path1 = os.path.join(output_folder, 'undistorted_video.avi')
    output_path2 = os.path.join(output_folder, 'origin_video.avi')
    # Dedistorted video
    file_undistorted_video = cv2.VideoWriter(output_path1, fourcc, 20.0, (640, 480))
    # orginal video
    file_origin_video = cv2.VideoWriter(output_path2, fourcc, 20.0, (640, 480))

    while inputVideo.isOpened():
        ret, oriframe = inputVideo.read()
        if not ret:
            break

        # The video frame is dedistorted
        undistorted_frame = cv2.fisheye.undistortImage(oriframe, camera_matrix, dist_coeff,  None, camera_matrix)

        # save
        file_undistorted_video.write(undistorted_frame)
        file_origin_video.write(oriframe)

        # show
        cv2.imshow("Original Video", oriframe)
        cv2.imshow("Undistorted Video", undistorted_frame)

        # press q to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Releases the video stream, closes the window, and releases the video save object
    inputVideo.release()
    file_undistorted_video.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    print("Please select the action: ")
    print("[1] Capture Image [2] Calibrate Camera ")
    selec = input()
    if selec == '1':
        # The captured video is stored in video
        CaptureVideo("video")
    elif selec == '2':
        # Captured video is stored in video and undistortedVideo
        undistort_video("video_and_undistortedVideo")