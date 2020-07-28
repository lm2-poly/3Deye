import cv2
import numpy as np
from calibration.chessboard_tools import get_chessboard_points


def get_cam_matrix(calib_pics, chess_dim, chess_case_len):
    """
    Finds the camera intrinsinc and distortion matrices

    :param calib_pics: calibration chessboard pictures
    :param chess_dim: Number of chess cases -1
    :param chess_case_len: chess case length
    :param pic: default None, picture frame to print in the gui
    :return: camera intrinsinc and distortion matrices
    """
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 1.e-9)
    objpoints, imgpoints, gray, img, objp, corners2 = get_chessboard_points(calib_pics, True, criteria, chess_dim)
    objpoints = chess_case_len * np.array(objpoints)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    return mtx, dist
