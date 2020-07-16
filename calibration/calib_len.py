import cv2
import numpy as np
from calibration.chessboard_tools import get_chessboard_points


def get_cam_matrix(calib_pics, pic=None):
    """
    Finds the camera intrinsinc and distortion matrices

    :param calib_pics: calibration chessboard pictures
    :return: camera intrinsinc and distortion matrices
    """
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objpoints, imgpoints, gray, img, objp, corners2 = get_chessboard_points(calib_pics, True, criteria, pic)
    objpoints = 0.4375 * np.array(objpoints)
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
