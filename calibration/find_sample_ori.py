import cv2
import numpy as np
from calibration.chessboard_tools import get_chessboard_points
from PIL import Image
import matplotlib.pyplot as plt


def get_transfo_mat(calib_pic_file, mtx, dist, pic=None):
    """
    Finds the transformation matrix between the camera's frame and the sample

    :param calib_pic_file: picture file of the chessboard
    :param mtx: camera intrinsinc matrix
    :param dist: camera distorsion matrix
    :return:
    """
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objpoints, imgpoints, gray, img, objp, corners2 = get_chessboard_points(calib_pic_file, False, criteria)
    objpoints = 0.4375 * np.array(objpoints)
    obj_vect = np.reshape(objpoints, (objpoints.shape[0] * objpoints.shape[1], 3))
    img_vect = np.reshape(np.array(imgpoints), (objpoints.shape[0] * objpoints.shape[1], 2))
    ret, R, T = cv2.solvePnP(obj_vect, img_vect, mtx, dist)
    return R, T


def plot_proj_origin(chessPic, mtx, dist, R, T, ax, name):
    pic = Image.open(chessPic)
    Rmat = np.matrix(np.zeros((3,3)))
    cv2.Rodrigues(R, Rmat)
    mat_pass = np.matrix(np.zeros((3, 4)))
    mat_pass[:, :3] = Rmat
    mat_pass[:, 3] = T

    ax.imshow(pic, cmap="gray")

    x_grid = np.linspace(0., 6 * 0.4375, 7)
    y_grid = np.linspace(0., 6 * 0.4375, 7)
    for i in range(0, 7):
        for j in range(0, 7):
            if name == 'top':
                vec_3D = np.matrix([x_grid[i], y_grid[j], 0, 1]).T
            else:
                vec_3D = np.matrix([-y_grid[i], x_grid[j], 0, 1]).T
            proj = mtx * mat_pass * vec_3D
            ax.plot([float(proj[0] / proj[2])], [float(proj[1] / proj[2])], '.', color="blue")

    vec_3D = np.matrix([0, 0, 0, 1]).T
    proj = mtx * mat_pass * vec_3D
    ax.plot([float(proj[0]/proj[2])], [float(proj[1]/proj[2])], '.', color="red", label='Axis origin')
    ax.legend()