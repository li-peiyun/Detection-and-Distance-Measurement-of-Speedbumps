import cv2
import numpy as np
imgps=np.array([[357,403],[487,409],[451,360],[358,355],[232,395],[264,353]])
objps=np.array([[0, 178.1],[60.2,178.1], [60.2,238.3],[0,238.3], [-60.2,178.1], [-60.2,238.3]])



if __name__ == '__main__':

    H, _ = cv2.findHomography(imgps, objps)
    np.save('configs/H_matrix.npy', H)
    