"""functions to reconstruct the object 3D trajectory"""
import numpy as np
from scipy.optimize import least_squares
from data_treat.objectExtract import compute_2d_traj
import matplotlib.pyplot as plt


def reconstruct_3d(cam_top, cam_left, splitSymb="_", numsplit=1, method="no-persp", plotTraj=True):
    """Reconstruct the 3D trajectory of a moving object filmed by 2 cameras with a given angle between them

    :param cam_top,cam_left: camera object for the top and left camera
    :param splitSymb: split symbol used in the images name between name and image number
    :param numsplit: index corresponding to the image number after the name was split
    :param method: "persp", "persp-opti" or "no-persp" (default) - using the
    camera intrinsinc matrix for 3D trajectory or not, using analytical expression
    or least square optimisation
    :param plotTraj: Boolean, if True the detected shot position will be plotted
    :return:
    """

    print("**** Shot position detection")
    print("** Top camera")
    traj_2d_top, timespan_top = compute_2d_traj(cam_top, splitSymb=splitSymb, numsplit=numsplit, plotTraj=plotTraj)
    print("** Left camera")
    traj_2d_left, timespan_left = compute_2d_traj(cam_left, splitSymb=splitSymb, numsplit=numsplit, plotTraj=plotTraj)
    minspan_len = min(len(timespan_top), len(timespan_left))

    traj_2d_top, traj_2d_left = cam_shift_resize(traj_2d_top, traj_2d_left, cam_top, cam_left)

    print("**** 3D position reconstruction")
    if method == "no-persp":
        X, Y, Z = get_3d_nopersp(minspan_len, traj_2d_left, traj_2d_top, cam_left, cam_top)
    else:
        X, Y, Z = get_3d_coor(minspan_len, traj_2d_left, traj_2d_top, cam_left, cam_top, method)
    plot_proj_error(traj_2d_top, traj_2d_left, X, Y, Z, cam_top, cam_left)

    return X, Y, Z, timespan_top[:minspan_len]


def cam_shift_resize(traj_2d_top, traj_2d_left, cam_top, cam_left):
    """Gives the 2D screen trajectory in the unresized picture frame

    :param traj_2d_top, traj_2d_left: screen trajectory for the top and left cameras
    :param cam_top,cam_left: top and left camera objects
    :return: shift_2d_top, shift_2d_left trajectory list
    """
    shift_2d_top = []
    shift_2d_left = []
    lenTraj = len(traj_2d_top)
    for i in range(0, lenTraj):
        x_t = traj_2d_top[i,0] + cam_top.camRes[0]/2 - cam_top.res[0]/2
        y_t = traj_2d_top[i, 1] + cam_top.camRes[1] / 2 - cam_top.res[1] / 2
        shift_2d_top.append([x_t, y_t])
        x_l = traj_2d_left[i, 0] + cam_left.camRes[0] / 2 - cam_left.res[0] / 2
        y_l = traj_2d_left[i, 1] + cam_left.camRes[1] / 2 - cam_left.res[1] / 2
        shift_2d_left.append([x_l, y_l])
    return np.array(shift_2d_top), np.array(shift_2d_left)


def plot_proj_error(traj_top, traj_left, X, Y, Z, cam_top, cam_left):
    """Plot the reprojected trajectory for each camera to check for the trajectory errors

        :param traj_top, traj_left: screen trajectory for the top and left cameras
        :param X,Y,Z: computed 3D trajectory
        :param cam_top,cam_left: top and left camera objects
        :return:
        """
    x_top, y_top = get_proj_list(X, Y, Z, cam_top)
    x_left, y_left = get_proj_list(-Y, Z, -X, cam_left)

    plt.figure(figsize=(18, 6))
    plt.title("Trajectory reprojection error")
    plt.subplot(131)
    plt.title("Top camera")
    plt.plot(traj_top[:, 0], traj_top[:, 1], label="Camera trajectory")
    plt.plot(x_top - cam_top.origin[0], y_top - cam_top.origin[1],'.', label="Reprojected trajectory")
    plot_square(cam_top)
    #plt.xlim((0, 1240))
    #plt.ylim((0, 800))
    plt.legend()
    plt.subplot(132)
    plt.title("Left camera")
    plt.plot(traj_left[:, 0], traj_left[:, 1], label="Camera trajectory")
    plt.plot(x_left- cam_left.origin[0], y_left- cam_left.origin[1], '.', label="Reprojected trajectory")
    plot_square(cam_left)
    plt.legend()
    plt.subplot(133)
    plt.title("Left camera")
    plt.plot(X, label="X")
    plt.plot(Y, label="Y")
    plt.plot(Z, label="Z")
    plt.legend()
    plt.show()
    #plt.xlim((0, 1240))
    #plt.ylim((0, 800))
    plt.show()


def plot_square(cam):
    A = [cam.camRes[0] / 2 - cam.res[0] / 2, cam.camRes[1] / 2 - cam.res[1] / 2]
    B = [cam.camRes[0] / 2 + cam.res[0] / 2, cam.camRes[1] / 2 - cam.res[1] / 2]
    C = [cam.camRes[0] / 2 + cam.res[0] / 2, cam.camRes[1] / 2 + cam.res[1] / 2]
    D = [cam.camRes[0] / 2 - cam.res[0] / 2, cam.camRes[1] / 2 + cam.res[1] / 2]
    plt.plot([A[0], B[0]], [A[1], B[1]], color="black")
    plt.plot([B[0], C[0]], [B[1], C[1]], color="black")
    plt.plot([C[0], D[0]], [C[1], D[1]], color="black")
    plt.plot([D[0], A[0]], [D[1], A[1]], color="black")


def get_proj_list(X, Y, Z, cam):
    """Computes the projection of a list of points in the 3D space onto the camera

    :param X,Y,Z: list of 3D coordinates
    :param cam: cam object
    :return two_lists x and y with the camera 2D projected oordinates"""
    lenList = X.shape[0]
    x = []
    y = []
    for i in range(0, lenList):
        dat_actu = get_proj_coords(X[i], Y[i], Z[i], cam)
        x.append(float(dat_actu[0] / dat_actu[2]))
        y.append(float(dat_actu[1] / dat_actu[2]))
    return np.array(x), np.array(y)


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
    return X_coor, Y_coor, Z_coor


def cam_shift_origin(cam):
    """Shift cam origin from the actual one to the cam.origin pixel value

    :param cam: camera object to tranform
    """
    actu_ori = get_proj_coords(0, 0, 0, cam)
    tx = cam.origin[0] - actu_ori[0]/actu_ori[2]
    ty = cam.origin[1] - actu_ori[1]/actu_ori[2]
    cam.origin = (float(tx), float(ty))


def make_system_mat(cam_top, cam_left, pos_2d_left, pos_2d_top):
    """Computes the matrix A, b of the equation system AX = b where X is the shot 3D coordinates

    :param pos_2d_left: position found by the left camera
    :param pos_2d_top: position found by the right camera
    :param cam_top,cam_left: camera object for the top and left camera
    :return:
    """

    A = np.zeros((3, 3))
    B = np.zeros((1, 3))
    u1, v1 = pos_2d_top + cam_top.origin
    u2, v2 = pos_2d_left + cam_left.origin
    a_top, b_top = make_alpha_beta(cam_top)
    a_left, b_left = make_alpha_beta(cam_left)

    A[0, :] = [cam_top.R[2, 0] * u1 - a_top[0, 0], cam_top.R[2, 1] * u1 - a_top[0, 1],
               cam_top.R[2, 2] * u1 - a_top[0, 2]]

    A[1, :] = [cam_top.R[2, 0] * v1 - a_top[1, 0], cam_top.R[2, 1] * v1 - a_top[1, 1],
                cam_top.R[2, 2] * v1 - a_top[1, 2]]

    #A[1, :] = [-cam_left.R[2, 2] * u2 + a_left[0, 2], -cam_left.R[2, 0] * u2 + a_left[0, 0],
    #            cam_left.R[2, 1] * u2 - a_left[0, 1]]
    A[2, :] = [-cam_left.R[2, 2] * v2 + a_left[1, 2], -cam_left.R[2, 0] * v2 + a_left[1, 0],
               cam_left.R[2, 1] * v2 - a_left[1, 1]]

    B[0, 0] = b_top[0, 0] - cam_top.T[2] * u1
    B[0, 1] = b_top[0, 1] - cam_top.T[2] * v1
    #B[0, 1] = b_left[0, 0] - cam_left.T[2] * u2
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
    pos_proj_left = get_proj_coords(-Y, Z, -X, cam_left)

    if pos_proj_top.T[0, 2] == 0.:
        top_uv = pos_proj_top.T[0, :2] - np.array([cam_top.origin[0], cam_top.origin[1]])
    else:
        top_uv = pos_proj_top.T[0, :2]/pos_proj_top.T[0, 2] - np.array([cam_top.origin[0], cam_top.origin[1]])
    if pos_proj_top.T[0, 2] == 0.:
        left_uv = pos_proj_left.T[0, :2]- np.array([cam_left.origin[0], cam_left.origin[1]])
    else:
        left_uv = pos_proj_left.T[0, :2] / pos_proj_left.T[0, 2] - np.array([cam_left.origin[0], cam_left.origin[1]])
    top_uv = np.reshape(np.array(top_uv), (2,))
    left_uv = np.reshape(np.array(left_uv), (2,))

    return np.reshape(np.array([top_uv - pos_2d_top, left_uv - pos_2d_left]), (4,))
