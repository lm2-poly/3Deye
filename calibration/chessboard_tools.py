"""
Various usefull function for chessboard calibration
"""
import cv2
import numpy as np
import glob


def order_points(pts):
    """Returns an ordered list of points

    :param pts: initial list of points
    """
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def get_chessboard_points(picDir, listPic, criteria, chess_dim):
    """Finds the chessboard points to use in camera calibration function

    :param picDir: chessboard picture directory path or single picture name (tif and jpg supported)
    :param listPic: True if picDir is the path to a list of pictures, False if it is a single picture name
    :param criteria: Chessboard position CV2 convergence criteria
    :param pic: default None, picture frame to print in the gui
    :return: objpoints (checkboard coordinates), imgpoints (picture coordinates), gray (chess gray picture), img, objp, corners2
    """

    objp = np.zeros((chess_dim * chess_dim, 3), np.float32)
    objp[:, :2] = np.mgrid[0:chess_dim, 0:chess_dim].T.reshape(-1, 2)

    objpoints = []
    imgpoints = []

    if listPic:
        images = glob.glob(picDir+'/*.tif')
        images += glob.glob(picDir+'/*.jpg')
    else:
        images = [picDir]

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (chess_dim, chess_dim), None)

        if ret == True:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

    return objpoints, imgpoints, gray, img, objp, corners2


def get_blob_position(img, corners, dim):
    """Find the grid position of the chessboard circles required to get the chessboard orientation

    :param img: chessboard picture
    :param corners: detected corner list
    :param dim: chessboard dimension
    :return: blob position in index coordinates
    """
    point_grid = np.reshape(corners, (dim, dim, 2))
    params = cv2.SimpleBlobDetector_Params()
    params.filterByCircularity = True
    params.minCircularity = 0.8
    params.minThreshold = 40
    params.filterByColor = 0
    detector = cv2.SimpleBlobDetector_create(params)
    blob_pos = []

    for i in range(0, dim-1):
        for j in range(0, dim-1):
            # x_min = int(min(point_grid[i, j, 0], point_grid[i,j+1,0]))
            # x_max = int(max(point_grid[i, j, 0], point_grid[i, j + 1, 0]))
            # y_min = int(min(point_grid[i, j, 1], point_grid[i+1, j, 1]))
            # y_max = int(max(point_grid[i, j, 1], point_grid[i+1, j, 1]))
            x_min = int(min(point_grid[i, j, 0], point_grid[i + 1, j, 0]))
            x_max = int(max(point_grid[i, j + 1, 0], point_grid[i + 1, j + 1, 0]))
            y_min = int(min(point_grid[i, j, 1], point_grid[i, j + 1, 1]))
            y_max = int(max(point_grid[i, j + 1, 1], point_grid[i + 1, j + 1, 1]))

            contour = np.array([point_grid[i, j], point_grid[i + 1, j], point_grid[i + 1, j + 1], point_grid[i, j + 1]])
            img_copy = mask_pic(contour, img)
            pic_actu = img_copy[y_min:y_max, x_min:x_max, :].astype('uint8')

            blob = detector.detect(pic_actu)
            if not(blob == []):
                blob_pos.append([j, i, pic_actu[int(blob[0].pt[1]), int(blob[0].pt[0])][0]])
    return blob_pos


def mask_pic(contour, img):
    img_copy = np.zeros(img.shape)
    lenx = img.shape[0]
    leny = img.shape[1]
    xmin = int(np.min(contour[:, 0]))
    xmax = int(np.max(contour[:, 0]))
    ymin = int(np.min(contour[:, 1]))
    ymax = int(np.max(contour[:, 1]))

    for i in range(xmin, xmax):
        for j in range(ymin, ymax):
            if not(cv2.pointPolygonTest(contour, (i, j), False) == -1.):
                img_copy[j, i] = img[j, i]
    return img_copy


def change_chess_ori(blobs, objpoints):
    """ Turns the object point array according to the detected blobs position

    :param blobs: Detected blobs positions
    :param objpoints: object point array
    :return: transformed object point array
    """
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

