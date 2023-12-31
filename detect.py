from ultralytics import YOLO
import cv2
from utils.split_video_and_undistort import split_video_to_undistroted_frames
from utils.generate_video import generate_video_from_frames
import numpy as np


# Video Distortion Correction
video_path = "video_and_undistortedVideo/origin_video3.avi"  # 需要根据需要修改，路径是一个原始视频
split_video_to_undistroted_frames(video_path)

# Loading the YOLOv8 model I trained
model = YOLO("speedbump.pt")

# Display only speed bumps: Speed bump corresponds to class=1
# Do not show labels, do not show confidence scores
# Source can be an image, a video, or a folder

# test data
#results = model.predict(source="test_images", save=False, classes=1, show_labels=False, show_conf=False)

# real data
results = model.predict(source="undistorted_image", save=False, classes=1, show_labels=False, show_conf=False)

# 'result' is the detection result for one image
# 'results' contains detection results for all images
# 'index' is the current iteration count
for index, result in enumerate(results):
    # Get the image corresponding to this detection result
    path = result.path
    # print(path)
    image = cv2.imread(path)

    # Iterate through all detection boxes (speed bumps) in the image
    for xyxy in result.boxes.xyxy:
        # 'corners' stores the corner coordinates of a speed bump in the format [xmin, ymin, xmax, ymax]
        corners = xyxy.tolist()

        # Extract vertex coordinates from 'corners'
        xmin, ymin, xmax, ymax = map(int, corners)

        # Draw a rectangle on the image
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)  # (0, 255, 0) 是绿色，2 是矩形的边框宽度

        # Calculate the distance of the speed bump
        xcenter = (xmin+xmax)/2
        ycenter = (ymin + ymax) / 2
        loaded_H = np.load('configs/H_matrix.npy')
        point_to_transform = np.array([[xcenter, ycenter]], dtype=np.float32)

        # Apply homography matrix to the point
        transformed_point = cv2.perspectiveTransform(point_to_transform.reshape(1, -1, 2), loaded_H)
        # Get the y-coordinate of the mapped point
        y_coordinate_transformed = transformed_point[0][0][1]

        distance = y_coordinate_transformed

        # Add text above the rectangle
        text = "distance=" + str(distance)
        cv2.putText(image, text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the image with rectangles
    # cv2.imshow("Image with Bounding Boxes", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Save the marked image in the 'marked_images' folder
    save_path = "marked_images\\" + str(index) + ".jpg"
    cv2.imwrite(save_path, image)

# Save the images with detected and measured speed bumps in the 'marked_images' folder

# Combine the images in the 'marked_images' folder into a video
output_video_path = "./generated_video/generated_video.mp4"
generate_video_from_frames(output_video_path, 20.0)
