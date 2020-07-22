import cv2
import numpy as np
from calibration.chessboard_tools import get_chessboard_points


def get_cam_matrix(calib_pics, chess_dim, chess_case_len, pic=None):
    """
    Finds the camera intrinsinc and distortion matrices

    :param calib_pics: calibration chessboard pictures
    :return: camera intrinsinc and distortion matrices
    """
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 1.e-9)
    objpoints, imgpoints, gray, img, objp, corners2 = get_chessboard_points(calib_pics, True, criteria, chess_dim, pic)
    objpoints = chess_case_len * np.array(objpoints)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1., (w, h))

    # undistort
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    # crop the image
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]
    # cv2.imwrite('res/calibresult.png', dst)
    # cv2.imwrite('res/initial.png', img)
    # cv2.imshow("distorted", img)
    # cv2.imshow("corrected", dst)
    # cv2.waitKey(500)

    return mtx, dist
