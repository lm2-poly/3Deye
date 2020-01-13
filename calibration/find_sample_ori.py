import cv2
import numpy as np
from calibration.chessboard_tools import get_chessboard_points


def get_transfo_mat(calib_pic_file, mtx, dist):
    """
    Finds the transformation matrix between the camera's frame and the sample

    :param calib_pic_file: picture file of the chessboard
    :param mtx: camera intrinsinc matrix
    :param dist: camera distorsion matrix
    :return:
    """
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.000001)

    objpoints, imgpoints, gray, img, objp, corners2 = get_chessboard_points(calib_pic_file, False, criteria)
    objpoints = 0.25 * np.array(objpoints)
    obj_vect = np.reshape(objpoints, (objpoints.shape[0] * objpoints.shape[1], 3))
    img_vect = np.reshape(np.array(imgpoints), (objpoints.shape[0] * objpoints.shape[1], 2))
    ret, R, T = cv2.solvePnP(obj_vect, img_vect, mtx, dist)
    return R, T
