"""functions to reconstruct the object 3D trajectory"""
import numpy as np
from scipy.optimize import least_squares
from objectExtract import compute_2d_traj


def reconstruct_3d(cam_top, cam_left, splitSymb="_", numsplit=1, method="no-persp"):
    """Reconstruct the 3D trajectory of a moving object filmed by 2 cameras with a given angle between them

    :param cam_top,cam_left: camera object for the top and left camera
    :param splitSymb: split symbol used in the images name between name and image number
    :param numsplit: index corresponding to the image number after the name was split
    :param method: "persp", "persp-opti" or "no-persp" (default) - using the
    camera intrinsinc matrix for 3D trajectory or not, using analytical expression
    or least square optimisation
    :return:
    """

    traj_2d_top, timespan_top = compute_2d_traj(cam_top, splitSymb=splitSymb, numsplit=numsplit)
    traj_2d_left, timespan_left = compute_2d_traj(cam_left, splitSymb=splitSymb, numsplit=numsplit)
    minspan_len = min(len(timespan_top), len(timespan_left))

    if method == "no-persp":
        X, Y, Z = get_3d_nopersp(minspan_len, traj_2d_left, traj_2d_top, cam_left, cam_top)
    else:
        X, Y, Z = get_3d_coor(minspan_len, traj_2d_left, traj_2d_top, cam_left, cam_top, method)
    return X, Y, Z, timespan_top[:minspan_len]


def get_3d_nopersp(minspan_len, traj_2d_left, traj_2d_top, cam_left, cam_top):
    """Find the 3D trajectory of the ball by orthographic projection nor camera exact positions

    :param minspan_len: number of time points
    :param traj_2d_left: trajectory found by the left camera
    :param traj_2d_top: trajectory found by the right camera
    :param cam_top,cam_left: camera object for the top and left camera
    :return:
    """
    X = (traj_2d_left[:minspan_len, 0] - 0.5 * cam_left.res[0]) * cam_left.pic_to_cm
    Y = (traj_2d_top[:minspan_len, 0] - 0.5 * cam_top.res[0]) * cam_top.pic_to_cm
    Z = (traj_2d_left[:minspan_len, 1] - 0.5 * cam_left.res[1]) * cam_left.pic_to_cm
    return X, Y, -Z


def get_3d_coor(minspan_len, traj_2d_left, traj_2d_top, cam_left, cam_top, method="persp"):
    """Retrieve the ball 3D trajectory from each cameras parameters and 2D trajectories

    :param minspan_len: number of time points
    :param traj_2d_left: trajectory found by the left camera
    :param traj_2d_top: trajectory found by the right camera
    :param cam_top,cam_left: camera object for the top and left camera
    :param method: "persp" (default) or "persp-opti" - use analytical expression
    or least square optimisation
    :return:
    """
    X = traj_2d_left[:minspan_len, 0]
    Y = traj_2d_top[:minspan_len, 0]
    Z = traj_2d_left[:minspan_len, 1]
    X_coor = np.zeros((minspan_len))
    Y_coor = np.zeros((minspan_len))
    Z_coor = np.zeros((minspan_len))
    for i in range(0, minspan_len):
        if not(np.isnan(X[i]) or np.isnan(Y[i]) or np.isnan(Z[i])):
            A, B = make_system_mat(cam_top, cam_left, traj_2d_left[i, :], traj_2d_top[i, :])
            X_coor[i], Y_coor[i], Z_coor[i] = np.linalg.solve(np.matrix(A), np.matrix(B).T)
            # X_coor[i], Y_coor[i], Z_coor[i] = np.linalg.pinv(np.matrix(A)) * np.matrix(B).T
            if method == "persp-opti":
                X0 = np.linalg.solve(np.matrix(A), np.matrix(B).T)
                args = (cam_left, cam_top, traj_2d_left[i, :], traj_2d_top[i, :])
                res_act = least_squares(get_proj_error, np.array(X0.T)[0], args=args)
                X_coor[i], Y_coor[i], Z_coor[i] = res_act.x
        else:
            X_coor[i], Y_coor[i], Z_coor[i] = [np.nan, np.nan, np.nan]
    return Y_coor, X_coor, -Z_coor


def make_system_mat(cam_top, cam_left, pos_2d_left, pos_2d_top):
    """Computes the matrix A, b of the equation system AX = b where X is the shot 3D coordinates

    :param pos_2d_left: position found by the left camera
    :param pos_2d_top: position found by the right camera
    :param cam_top,cam_left: camera object for the top and left camera
    :return:
    """
    A = np.zeros((3, 3))
    B = np.zeros((1, 3))
    u1, v1 = pos_2d_top
    u2, v2 = pos_2d_left
    a_top, b_top = make_alpha_beta(cam_top)
    a_left, b_left = make_alpha_beta(cam_left)

    A[0, :] = [cam_top.R[2, 0] * u1 - a_top[0, 0], cam_top.R[2, 1] * u1 - a_top[0, 1],
               cam_top.R[2, 2] * u1 - a_top[0, 2]]

    # A[1, :] = [cam_top.R[2, 0] * v1 - a_top[1, 0], cam_top.R[2, 1] * v1 - a_top[1, 1],
    #            cam_top.R[2, 2] * v1 - a_top[1, 2]]

    A[1, :] = [cam_left.R[2, 2] * u2 - a_left[0, 2], cam_left.R[2, 0] * u2 - a_left[0, 0],
                cam_left.R[2, 1] * u2 - a_left[0, 1]]
    A[2, :] = [cam_left.R[2, 2] * v2 - a_left[1, 2], cam_left.R[2, 0] * v2 - a_left[1, 0],
               cam_left.R[2, 1] * v2 - a_left[1, 1]]

    B[0, 0] = b_top[0, 0] - cam_top.T[2] * u1
    # B[0, 1] = b_top[0, 1] - cam_top.T[2] * v1
    B[0, 1] = b_left[0, 0] - cam_left.T[2] * u2
    B[0, 2] = b_left[0, 1] - cam_left.T[2] * v2
    return A, B


def make_alpha_beta(cam):
    """Compute necessary coefficients for the 3D coordinate system matrix AX=b
    for better code readability.

    :param cam: camera object
    """
    fx = cam.mtx[0, 0]
    fy = cam.mtx[1, 1]
    cx = cam.mtx[0, 2]
    cy = cam.mtx[1, 2]
    R = cam.R
    alpha = np.zeros((2, 3))
    alpha[0, 0] = fx * R[0, 0] + cx * R[2, 0]
    alpha[0, 1] = fx * R[0, 1] + cx * R[2, 1]
    alpha[0, 2] = fx * R[0, 2] + cx * R[2, 2]
    alpha[1, 0] = fy * R[1, 0] + cy * R[2, 0]
    alpha[1, 1] = fy * R[1, 1] + cy * R[2, 1]
    alpha[1, 2] = fy * R[1, 2] + cy * R[2, 2]

    beta = np.zeros((1, 2))
    beta[0, 0] = fx * cam.T[0] + cx * cam.T[2]
    beta[0, 1] = fy * cam.T[1] + cy * cam.T[2]
    return alpha, beta


def get_proj_coords(X, Y, Z, cam):
    """Compute the projection of 3D coordinates on a camera screen.

    :param X,Y,Z: 3D coordinates to be projected
    :param cam: camera object used for projection
    :return:
    """
    mat_pass = np.matrix(np.zeros((3,4)))
    mat_pass[:, :3] = cam.R
    mat_pass[:, 3] = np.matrix(cam.T).T
    vec_3D = np.matrix([X, Y, Z, 1]).T
    return cam.mtx * mat_pass * vec_3D


def get_proj_error(var, cam_left, cam_top, pos_2d_left, pos_2d_top):
    """Compute the projection error between the projection of
     a guessed 3D coordinate vector and the actual screen point

    :param var: 3D coordinate triplet
    :param pos_2d_left: position found by the left camera
    :param pos_2d_top: position found by the right camera
    :param cam_top,cam_left: camera object for the top and left camera
    :return:
    """
    X, Y, Z = var
    pos_proj_top = get_proj_coords(X, Y, Z, cam_top)
    pos_proj_left = get_proj_coords(Y, Z, X, cam_left)

    if pos_proj_top.T[0, 2] == 0.:
        top_uv = pos_proj_top.T[0, :2]
    else:
        top_uv = pos_proj_top.T[0, :2]/pos_proj_top.T[0, 2]
    if pos_proj_top.T[0, 2] == 0.:
        left_uv = pos_proj_left.T[0, :2]
    else:
        left_uv = pos_proj_left.T[0, :2] / pos_proj_left.T[0, 2]
    top_uv = np.reshape(np.array(top_uv), (2,))
    left_uv = np.reshape(np.array(left_uv), (2,))

    return np.reshape(np.array([top_uv - pos_2d_top, left_uv - pos_2d_left]), (4,))