import cv2
import numpy as np
imgps=[]
def click_corner(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        cv2.circle(img, (x, y), 5, (255, 0, 0), thickness=-1)
        cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness=1)
        imgps.append([x, y])

if __name__ == '__main__':
    img = cv2.imread('TileImage/UndistortionTileImg2/1_undistorted.png')
    cv2.destroyAllWindows()
    cv2.namedWindow("groundBoard")
    cv2.setMouseCallback("groundBoard", click_corner)

    while (1):
        cv2.imshow("groundBoard", img)
        key = cv2.waitKey(1) & 0xff
        if key == ord('q') or key == ord('Q'):
            imgps = np.array(imgps, dtype=np.float32)  # change type to np.ndarray
            print(imgps)
            output_path = 'D:/Detection-and-Distance-Measurement-of-Speedbumps-main/HomoImage/calibrated_image.png'
            cv2.imwrite(output_path, img)
            break