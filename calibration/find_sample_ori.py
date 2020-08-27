import cv2
import numpy as np
from calibration.chessboard_tools import get_chessboard_points, get_blob_position, change_chess_ori
from PIL import Image
import matplotlib.pyplot as plt


def get_transfo_mat(calib_pic_file, mtx, dist, chess_dim, chess_case_len):
    """
    Finds the transformation matrix between the camera's frame and the sample

    :param calib_pic_file: picture file of the chessboard
    :param mtx: camera intrinsinc matrix
    :param dist: camera distorsion matrix
    :param chess_dim: Number of cases per chess side minus one
    :param chess_case_len: real life lenth of the chess square
    :return: R, T Rotation Rodrigues vector and Translation vector
    """
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 1.e-2)

    objpoints, imgpoints, gray, img, objp, corners2 = get_chessboard_points(calib_pic_file, False, criteria, chess_dim)
    objpoints = chess_case_len * np.array(objpoints)
    obj_vect = np.reshape(objpoints, (objpoints.shape[0] * objpoints.shape[1], 3))
    img_vect = np.reshape(np.array(imgpoints), (objpoints.shape[0] * objpoints.shape[1], 2))
    blobs = blobs = get_blob_position(img, img_vect, chess_dim)
    obj_vect = change_chess_ori(blobs, objpoints)

    ret, R, T = cv2.solvePnP(obj_vect, img_vect, mtx, dist)
    return R, T


def plot_proj_origin(chessPic, mtx, R, T, chess_dim, chess_case_len):
    """ Plots the chessboard point projection and chessboard coordinate system axis on a camera for visual calibration check

    :param chessPic: Chessboard picture path
    :param mtx: camera intrinsic matrix
    :param R: camera Rotation Rodrigues vector
    :param T: camera translation vector
    :param chess_dim: Number of cases per chess side minus one
    :param chess_case_len: real life lenth of the chess square
    """
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
            vec_3D = np.matrix([x_grid[i], y_grid[j], 0, 1]).T
            proj = mtx * mat_pass * vec_3D
            plt.plot([float(proj[0] / proj[2])], [float(proj[1] / proj[2])], '.', color="blue")
    ori, Xaxis = get_quiver([chess_case_len, 0., 0., 1.], mtx, mat_pass)
    ori, Yaxis = get_quiver([0., chess_case_len, 0., 1.], mtx, mat_pass)
    ori, Zaxis = get_quiver([0., 0., chess_case_len, 1.], mtx, mat_pass)

    plt.plot([float(proj[0] / proj[2])], [float(proj[1] / proj[2])], '.', color="blue", label="Projected point")
    vec_3D = np.matrix([0, 0, 0, 1]).T
    proj = mtx * mat_pass * vec_3D
    plt.plot([float(proj[0]/proj[2])], [float(proj[1]/proj[2])], '.', color="red", label='Axis origin')
    plt.quiver(ori[0], ori[1], Xaxis[0], Xaxis[1], color="red")
    plt.quiver(ori[0], ori[1], Yaxis[0], Yaxis[1], color="green")
    plt.quiver(ori[0], ori[1], Zaxis[0], Zaxis[1], color="blue")
    plt.xlim((0., pic.size[0]))
    plt.ylim((pic.size[1], 0.))
    plt.legend()


def get_quiver(axis, mtx, mat_pass):
    """ Computes the projection of a vector on a give camera

    :param axis: vector to project
    :param mtx: camera intrinsic matrix
    :param mat_pass: camera transformation matrix
    :return:
    """
    origin = np.matrix([0., 0., 0., 1.]).T
    proj_ori = mtx * mat_pass * origin
    proj_axis = mtx * mat_pass * np.matrix(axis).T
    X0 = float(proj_ori[0] / proj_ori[2])
    Y0 = float(proj_ori[1] / proj_ori[2])
    X1 = float(proj_axis[0] / proj_axis[2])
    Y1 = float(proj_axis[1] / proj_axis[2])
    return [X0, Y0], [X1 - X0, Y0 - Y1]