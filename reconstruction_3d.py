"""
functions to reconstruct the object 3D trajectory
"""
import numpy as np
from objectExtract import compute_2d_traj


def reconstruct_3d(cam_top, cam_left, splitSymb="_", numsplit=1):
    """
    Reconstruct the 3D trajectory of a moving object filmed by 2 cameras with a given angle between them

    :param cam_top:
    :param cam_left:
    :param splitSymb:
    :param numsplit:
    :return:
    """

    traj_2d_top, timespan_top = compute_2d_traj(cam_top, splitSymb="_", numsplit=1)
    traj_2d_left, timespan_left = compute_2d_traj(cam_left, splitSymb="_", numsplit=1)
    minspan_len = min(len(timespan_top), len(timespan_left))
    X, Y, Z = get_3d_coor(timespan_left, timespan_top, traj_2d_left, traj_2d_top, cam_left, cam_top)
    return X, Y, Z, timespan_top[:minspan_len]


def get_3d_coor(timespan_left, timespan_top, traj_2d_left, traj_2d_top, cam_left, cam_top):
    minspan_len = min(len(timespan_top), len(timespan_left))
    X = traj_2d_left[:minspan_len, 0]
    Y = traj_2d_top[:minspan_len, 0]
    Z = traj_2d_left[:minspan_len, 1]
    X_coor = np.zeros((minspan_len))
    Y_coor = np.zeros((minspan_len))
    Z_coor = np.zeros((minspan_len))
    coord_pres = [-0.5,1., 1.]
    for i in range(0, minspan_len):
        args = (cam_left, cam_top, traj_2d_left[i, :], traj_2d_top[i, :])
        X0 = [X[i], Y[i], Z[i]]
        if not(np.isnan(X[i]) or np.isnan(Y[i]) or np.isnan(Z[i])):
            A, B = make_system_mat(cam_top, cam_left, traj_2d_left[i, :], traj_2d_top[i, :])
            X_coor[i], Y_coor[i], Z_coor[i] = np.linalg.solve(np.matrix(A), np.matrix(B).T)
            #X_coor[i], Y_coor[i], Z_coor[i] = np.linalg.pinv(A)*np.matrix(B).T
            #X0 = np.linalg.solve(np.matrix(A), np.matrix(B).T)
            #res_act = least_squares(get_proj_error, np.array(X0.T)[0], args=args)
            #X_coor[i], Y_coor[i], Z_coor[i] = res_act.x
        else:
            X_coor[i], Y_coor[i], Z_coor[i] = [np.nan, np.nan, np.nan]
    return X_coor, Y_coor, Z_coor


def get_proj_error(var, cam_left, cam_top, pos_2d_left, pos_2d_top):
    X, Y, Z = var
    pos_proj_top = get_proj_coords(X, Y, Z, cam_top)
    pos_proj_left = get_proj_coords(-Y, Z, -X, cam_left)

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

def make_system_mat(cam_top, cam_left, pos_2d_left, pos_2d_top):
    A = np.zeros((3, 3))
    B = np.zeros((1, 3))
    u1, v1 = pos_2d_top
    u2, v2 = pos_2d_left
    a_top, b_top = make_alpha_beta(cam_top)
    a_left, b_left = make_alpha_beta(cam_left)

    A[0, :] = [cam_top.R[2, 0] * u1 - a_top[0, 0], cam_top.R[2, 1] * u1 - a_top[0, 1],
               cam_top.R[2, 2] * u1 - a_top[0, 2]]

    A[1, :] = [cam_top.R[2, 0] * v1 - a_top[1, 0], cam_top.R[2, 1] * v1 - a_top[1, 1],
               cam_top.R[2, 2] * v1 - a_top[1, 2]]

    # A[2, :] = [-cam_left.R[2, 2] * u2 + a_left[0, 2], -cam_left.R[2, 0] * u2 + a_left[0, 0],
    #            cam_left.R[2, 1] * u2 - a_left[0, 1]]
    A[2, :] = [-cam_left.R[2, 2] * v2 + a_left[1, 2], -cam_left.R[2, 0] * v2 + a_left[1, 0],
               cam_left.R[2, 1] * v2 - a_left[1, 1]]

    B[0, 0] = b_top[0, 0] - cam_top.T[2] * u1
    B[0, 1] = b_top[0, 1] - cam_top.T[2] * v1
    # B[0, 2] = b_left[0, 0] - cam_left.T[2] * u2
    B[0, 2] = b_left[0, 1] - cam_left.T[2] * v2
    return A, B


def make_alpha_beta(cam):
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
    mat_pass = np.matrix(np.zeros((3,4)))
    mat_pass[:, :3] = cam.R
    mat_pass[:, 3] = np.matrix(cam.T).T
    vec_3D = np.matrix([X, Y, Z, 1]).T
    return cam.mtx * mat_pass * vec_3D
