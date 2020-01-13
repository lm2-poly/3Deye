"""
... not really usefull but did not deleted as I can't remember what's in it
"""
import numpy as np
import cv2
import glob
import math

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 1.e-8)


def make_points_data(images, criteria):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((7 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:7, 0:9].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (7,9),None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (7,9), corners2,ret)
            cv2.imshow('img',img)
            cv2.waitKey(500)
    return objpoints, imgpoints, gray

images = glob.glob('../lens_dist_calib/*.png')
objpoints1, imgpoints1, gray1 = make_points_data(images, criteria)
ret1, K1, D1, rvecs1, tvecs1 = cv2.calibrateCamera(objpoints1, imgpoints1, gray1.shape[::-1],None,None)
images = glob.glob('../lens_dist_calib/*.png')
objpoints2, imgpoints2, gray2 = make_points_data(images, criteria)
ret2, K2, D2, rvecs2, tvecs2 = cv2.calibrateCamera(objpoints2, imgpoints2, gray2.shape[::-1],None,None)

images = glob.glob('../stereo_calib/Top/*.png')
objpoints1, imgpoints1, gray1 = make_points_data(images, criteria)
images = glob.glob('../stereo_calib/Left/*.png')
objpoints2, imgpoints2, gray2 = make_points_data(images, criteria)

retval, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(objpoints1, imgpoints1, imgpoints2, K1, D1, K2, D2, gray1.shape[::-1], criteria=criteria, flags=cv2.CALIB_FIX_INTRINSIC)
R_rod = cv2.Rodrigues(R)[0]
print(R_rod/np.linalg.norm(R_rod))
print(np.linalg.norm(R_rod)*180/math.pi)