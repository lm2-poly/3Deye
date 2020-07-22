import cv2
import numpy as np
from calibration.chessboard_tools import get_chessboard_points
from PIL import Image
import matplotlib.pyplot as plt


def get_transfo_mat(calib_pic_file, mtx, dist, chess_dim, chess_case_len, pic=None):
    """
    Finds the transformation matrix between the camera's frame and the sample

    :param calib_pic_file: picture file of the chessboard
    :param mtx: camera intrinsinc matrix
    :param dist: camera distorsion matrix
    :param chess_case_len: real life lenth of the chess square
    :return:
    """
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 1.e-9)

    objpoints, imgpoints, gray, img, objp, corners2 = get_chessboard_points(calib_pic_file, False, criteria, chess_dim,pic)
    objpoints = chess_case_len * np.array(objpoints)
    obj_vect = np.reshape(objpoints, (objpoints.shape[0] * objpoints.shape[1], 3))
    img_vect = np.reshape(np.array(imgpoints), (objpoints.shape[0] * objpoints.shape[1], 2))
    ret, R, T = cv2.solvePnP(obj_vect, img_vect, mtx, dist)
    return R, T


def plot_proj_origin(chessPic, mtx, dist, R, T, name, chess_dim, chess_case_len):
    pic = Image.open(chessPic)
    Rmat = np.matrix(np.zeros((3,3)))
    cv2.Rodrigues(R, Rmat)
    mat_pass = np.matrix(np.zeros((3, 4)))
    mat_pass[:, :3] = Rmat
    mat_pass[:, 3] = T

    plt.imshow(pic, cmap="gray")

    x_grid = np.linspace(0., (chess_dim-1) * chess_case_len, chess_dim)
    y_grid = np.linspace(0., (chess_dim-1) * chess_case_len, chess_dim)
    for i in range(0, chess_dim):
        for j in range(0, chess_dim):
            if name == 'top':
                vec_3D = np.matrix([x_grid[i], y_grid[j], 0, 1]).T
            else:
                vec_3D = np.matrix([-y_grid[i], x_grid[j], 0, 1]).T
            proj = mtx * mat_pass * vec_3D
            plt.plot([float(proj[0] / proj[2])], [float(proj[1] / proj[2])], '.', color="blue")
    plt.plot([float(proj[0] / proj[2])], [float(proj[1] / proj[2])], '.', color="blue", label="Projected point")
    vec_3D = np.matrix([0, 0, 0, 1]).T
    proj = mtx * mat_pass * vec_3D
    plt.plot([float(proj[0]/proj[2])], [float(proj[1]/proj[2])], '.', color="red", label='Axis origin')
    plt.legend()