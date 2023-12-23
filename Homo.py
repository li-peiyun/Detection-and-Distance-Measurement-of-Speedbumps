import cv2
import numpy as np
imgps=np.array([[357,403],[487,409],[451,360],[358,355],[232,395],[264,353]])
objps=np.array([[0, 178.1],[60.2,178.1], [60.2,238.3],[0,238.3], [-60.2,178.1], [-60.2,238.3]])
# imgps=[]
# def click_corner(event, x, y, flags, param):
#     if event == cv2.EVENT_LBUTTONDOWN:
#         xy = "%d,%d" % (x, y)
#         cv2.circle(img, (x, y), 5, (255, 0, 0), thickness=-1)
#         cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness=1)
#         imgps.append([x, y])


if __name__ == '__main__':

    # H, _ = cv2.findHomography(imgps, objps)
    # np.save('configs/H_matrix.npy', H)
    loaded_H = np.load('configs/H_matrix.npy')
    point_to_transform = np.array([[264, 353]], dtype=np.float32)

    # 应用单应性矩阵到点
    transformed_point = cv2.perspectiveTransform(point_to_transform.reshape(1, -1, 2), loaded_H)
    # 获取映射后点的 y 坐标
    y_coordinate_transformed = transformed_point[0][0][1]

    distance = y_coordinate_transformed  # 测试数据
    print("distance:", distance)
    # 打印结果
    # print("Original Point:", point_to_transform)
    # print("Transformed Point:", transformed_point[0][0])

    # img = cv2.imread('TileImage/UndistortionTileImg2/1_undistorted.png')
    # cv2.destroyAllWindows()
    # cv2.namedWindow("groundBoard")
    # cv2.setMouseCallback("groundBoard", click_corner)
    #
    # while (1):
    #     cv2.imshow("groundBoard", img)
    #     key = cv2.waitKey(1) & 0xff
    #     if key == ord('q') or key == ord('Q'):
    #         imgps = np.array(imgps, dtype=np.float32)  # change type to np.ndarray
    #         # objps = np.array(objps, dtype=np.float32)
    #         # H, _ = cv2.findHomography(imgps, objps)
    #         # cv2.waitKey(0)
    #         # print(H, type(H))
    #         print(imgps)
    #         output_path = 'D:/Detection-and-Distance-Measurement-of-Speedbumps-main/HomoImage/calibrated_image.png'
    #         cv2.imwrite(output_path, img)
    #         break