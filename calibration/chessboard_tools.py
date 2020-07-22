"""
Various usefull function for chessboard calibration
"""
import cv2
import numpy as np
import glob
import tkinter as tk
from PIL import Image, ImageTk


def draw(img, corners, imgpts):
    """
    draw chessboard corners

    :param img: chessboard picture
    :param corners: chessboard corner points
    :param imgpts: chessboard points
    :return:
    """

    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255, 0, 0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0, 0, 255), 5)
    return img


def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect


def get_chessboard_points(picDir, listPic, criteria, chess_dim, pic=None):
    """
    Finds the chessboard points to use in camera calibration function
    :param picDir:
    :param listPic:
    :param criteria:
    :return:
    """
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((chess_dim * chess_dim, 3), np.float32)
    objp[:, :2] = np.mgrid[0:chess_dim, 0:chess_dim].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    if listPic:
        images = glob.glob(picDir+'/*.jpg')
    else:
        images = [picDir]

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (chess_dim, chess_dim), None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            #img_corn = cv2.drawChessboardCorners(img, (chess_dim, chess_dim), corners2, ret)
            # if pic is None:
            #     cv2.imshow('img', img)
            #     cv2.waitKey(500)
            #     cv2.destroyAllWindows()

    return objpoints, imgpoints, gray, img, objp, corners2


def get_blob_position(img, corners, dim):
    """Find the grid position of the chessboard circles required to get the chessboard orientation

    :param img: chessboard picture
    :param corners: detected corner list
    :param dim: chessboard dimension
    """
    point_grid = np.reshape(corners, (dim, dim, 2))
    params = cv2.SimpleBlobDetector_Params()
    # params.minThreshold = 10;
    # params.maxThreshold = 200;
    params.filterByCircularity = True
    params.minCircularity = 0.8
    params.filterByColor = 0
    detector = cv2.SimpleBlobDetector_create(params)
    blob_pos = []

    for i in range(0, dim-1):
        for j in range(0, dim-1):
            x_min = int(min(point_grid[i, j, 0], point_grid[i,j+1,0]))
            x_max = int(max(point_grid[i, j, 0], point_grid[i, j + 1, 0]))
            y_min = int(min(point_grid[i, j, 1], point_grid[i+1, j, 1]))
            y_max = int(max(point_grid[i, j, 1], point_grid[i+1, j, 1]))
            pic_actu = img[y_min:y_max, x_min:x_max, :]
            blob = detector.detect(pic_actu)
            if not(blob == []):
                blob_pos.append([j, i, pic_actu[int(blob[0].pt[1]), int(blob[0].pt[0])][0]])
    return blob_pos


def change_chess_ori(blobs, objpoints):
    i = 0
    bcor = np.reshape(blobs, (len(blobs), 3))[:, :2]
    for blob in blobs:
        if blob[2] < 80:
            b_black = i
        i +=1

    tmp_points = np.copy(np.reshape(objpoints, (7, 7, 3)))
    tmp_obj = np.reshape(objpoints, (7, 7, 3))

    if np.sum(bcor[b_black] == [1, 3]) ==2:
        for i in range(0, 7):
            tmp_points[:, i, 0] = tmp_obj[:, i, 1]
            tmp_points[:, i, 1] = tmp_obj[:, 6-i, 0]
    elif np.sum(bcor[b_black] == [2, 1]) ==2:
        for i in range(0, 7):
            tmp_points[:, i, 0] = tmp_obj[:, 6 - i, 0]
            tmp_points[:, i, 1] = tmp_obj[:, 6 - i, 1]
    elif np.sum(bcor[b_black] == [4, 2]) ==2:
        for i in range(0, 7):
            tmp_points[i, :, 0] = tmp_obj[6 - i, :, 1]
            tmp_points[i, :, 1] = tmp_obj[i, :, 0]

    return np.reshape(tmp_points, (49, 3))

