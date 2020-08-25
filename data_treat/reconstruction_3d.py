"""functions to reconstruct the object 3D trajectory"""
import numpy as np
from scipy.optimize import least_squares, differential_evolution
from scipy.interpolate import interp1d
from data_treat.objectExtract import compute_2d_traj
import matplotlib.pyplot as plt
from gui.gui_utils import plot_fig
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
import glob
from PIL import Image
import tkinter as tk


def reconstruct_3d(cam_top, cam_left, splitSymb="_", numsplit=1, method="no-persp",
                   plotTraj=True, plot=True, isgui=False, savedir='data_treat/Reproj_error.png'):
    """Reconstruct the 3D trajectory of a moving object filmed by 2 cameras with a given angle between them

    :param cam_top,cam_left: camera object for the top and left camera
    :param splitSymb: split symbol used in the images name between name and image number
    :param numsplit: index corresponding to the image number after the name was split (default 1)
    :param method: "persp", "persp-opti" or "no-persp" (default) - using the camera intrinsinc matrix for 3D trajectory or not, using analytical expression or least square optimisation
    :param plotTraj: Boolean, if True the detected shot position will be plotted
    :param plot: Boolean, if true the reprojection error will be plotted
    :param savedir: path to the directory to save the reprojection error to
    :return:
    """

    print("**** Shot position detection")
    print("** Top camera")
    traj_2d_top, timespan_top = compute_2d_traj(cam_top, splitSymb=splitSymb, numsplit=numsplit, plotTraj=False, isgui=isgui)
    print("** Left camera")
    traj_2d_left, timespan_left = compute_2d_traj(cam_left, splitSymb=splitSymb, numsplit=numsplit, plotTraj=False, isgui=isgui)
    minspan_len = min(len(timespan_top), len(timespan_left))

    if plotTraj:
        plot_cam_traj(cam_top, traj_2d_top, "Top camera")
        plot_cam_traj(cam_left, traj_2d_left, "Left camera")

    traj_2d_top, traj_2d_left = cam_shift_resize(traj_2d_top, traj_2d_left, cam_top, cam_left)

    print("**** 3D position reconstruction")
    if method == "no-persp":
        X, Y, Z = get_3d_nopersp(minspan_len, traj_2d_left, traj_2d_top, cam_left, cam_top)
    else:
        X, Y, Z, traj_2d_top, traj_2d_left = get_3d_coor(minspan_len, traj_2d_left, traj_2d_top, cam_left, cam_top, method, timespan_top[:minspan_len])

    plot_proj_error(traj_2d_top, traj_2d_left, X, Y, Z, cam_top, cam_left, savedir=savedir, plot=plot)



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


def plot_proj_error(traj_top, traj_left, X, Y, Z, cam_top, cam_left, savedir='data_treat/Reproj_error.png', plot=True):
    """Plot the reprojected trajectory for each camera to check for the trajectory errors

        :param traj_top,traj_left: screen trajectory for the top and left cameras
        :param X,Y,Z: computed 3D trajectory
        :param cam_top,cam_left: top and left camera objects
        :param savedir: path to the directory to save the reprojection error to
        """
    x_top, y_top = get_proj_list(X, Y, Z, cam_top)
    x_left, y_left = get_proj_list(X, Y, Z, cam_left)

    fig = Figure(figsize=(18, 6), tight_layout=True)
    ax1, ax2, ax3 = fig.subplots(ncols=3, nrows=1)
    ax1.set_title("Top camera")
    ax1.plot(traj_top[:, 0], traj_top[:, 1], 'o-', label="Camera trajectory")
    ax1.plot(x_top, y_top,'.', label="Reprojected trajectory")
    plot_square(cam_top, ax1)
    ax1.legend()

    ax2.set_title("Left camera")
    ax2.plot(traj_left[:, 0], traj_left[:, 1], 'o-', label="Camera trajectory")
    ax2.plot(x_left, y_left, '.', label="Reprojected trajectory")
    plot_square(cam_left, ax2)
    ax2.legend()

    ax3.set_title("Shot 3D Trajectory")
    ax3.plot(X, label="X")
    ax3.plot(Y, label="Y")
    ax3.plot(Z, label="Z")
    ax3.set_xlabel("time (ms)")
    ax3.set_ylabel("Position (cm)")
    ax3.legend()
    #
    fig.savefig(savedir)
    if plot:
        root, canvas = plot_fig(fig, size="1200x450")

    #plt.show(block=False)


def plot_cam_traj(cam, traj, title):
    pic_num = tk.IntVar()
    picList = glob.glob(cam.dir + "/*.tif")
    if len(picList) == 0:
        picList = glob.glob(cam.dir + "/*.jpg")
    fig = Figure(tight_layout=True)
    root, canvas = plot_fig(fig, size="600x600")
    lab = tk.Label(root, text=title)
    lab.pack(side=tk.TOP)
    update_pic(0, canvas, cam, traj, picList)
    mask_val_w = tk.Scale(root, from_=0, to=len(picList), orient=tk.HORIZONTAL, variable=pic_num,
                          command=(lambda ma=pic_num, c=canvas, ca=cam, tr=traj, pl=picList: update_pic(ma,c, ca, tr, pl)))
    mask_val_w.pack(side=tk.BOTTOM)


def update_pic(pic_num, canvas, cam, traj, picList):
    img = get_treated_pic(cam, picList[int(pic_num)])
    canvas.figure.clear()
    canvas.figure.add_subplot(111).imshow(img, cmap='gray')
    canvas.figure.add_subplot(111).plot(traj[:, 1], traj[:, 0], '.', color='red', ms=1.5)
    canvas.draw()


def get_treated_pic(cam, pic):
    img = Image.open(pic).convert('LA')

    width, height = img.size
    RGBPicRef = (np.array(img)[:, :, 0].T).astype(np.int16)
    RGBPicRef[:int(cam.mask_w), :] = 0  # Width mask
    RGBPicRef[:, RGBPicRef.shape[1] - int(cam.mask_h):] = 0  # Height mask
    RGBPicRef = RGBPicRef[:, :height - cam.cropSize[3]]  # Vertical crop
    RGBPicRef = RGBPicRef[cam.cropSize[0]:width - cam.cropSize[1], :]  # Horizontal crop
    return RGBPicRef


def plot_square(cam, ax):
    """Plot a square repreentive the cropped camera screen

    :param cam: camera object
    """
    A = [cam.camRes[0] / 2 - cam.res[0] / 2, cam.camRes[1] / 2 - cam.res[1] / 2]
    B = [cam.camRes[0] / 2 + cam.res[0] / 2, cam.camRes[1] / 2 - cam.res[1] / 2]
    C = [cam.camRes[0] / 2 + cam.res[0] / 2, cam.camRes[1] / 2 + cam.res[1] / 2]
    D = [cam.camRes[0] / 2 - cam.res[0] / 2, cam.camRes[1] / 2 + cam.res[1] / 2]
    ax.plot([A[0], B[0]], [A[1], B[1]], color="black")
    ax.plot([B[0], C[0]], [B[1], C[1]], color="black")
    ax.plot([C[0], D[0]], [C[1], D[1]], color="black")
    ax.plot([D[0], A[0]], [D[1], A[1]], color="black")


def get_proj_list(X, Y, Z, cam):
    """Computes the projection of a list of points in the 3D space onto the camera

    :param X,Y,Z: list of 3D coordinates
    :param cam: cam object
    :return: two_lists x and y with the camera 2D projected coordinates
    """
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
    :return: X,Y,Z coordinate list
    """
    X = (traj_2d_top[:minspan_len, 0] - 0.5 * cam_left.res[0]) * cam_left.pic_to_cm
    Y = (traj_2d_top[:minspan_len, 1] - 0.5 * cam_top.res[0]) * cam_top.pic_to_cm
    Z = (traj_2d_left[:minspan_len, 1] - 0.5 * cam_left.res[1]) * cam_left.pic_to_cm
    return X, Y, Z


def get_3d_coor(minspan_len, traj_2d_left, traj_2d_top, cam_left, cam_top, method="persp", timespan=[]):
    """Retrieve the shot 3D trajectory from each cameras parameters and 2D trajectories

    :param minspan_len: number of time points
    :param traj_2d_left: trajectory found by the left camera
    :param traj_2d_top: trajectory found by the right camera
    :param cam_top,cam_left: camera object for the top and left camera
    :param method: "persp" (default) or "persp-opti" - use analytical expression or least square optimisation
    :return:
    """
    t1 = traj_2d_top
    t2 = traj_2d_left
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
            if method == "persp-opti-coor":
                X0 = np.linalg.solve(np.matrix(A), np.matrix(B).T)
                args = (cam_left, cam_top, traj_2d_left[i, :], traj_2d_top[i, :])
                res_act = least_squares(get_proj_error, np.array(X0.T)[0], args=args)
                X_coor[i], Y_coor[i], Z_coor[i] = res_act.x
        else:
            X_coor[i], Y_coor[i], Z_coor[i] = [np.nan, np.nan, np.nan]

    if method == "persp-opti":
        #tau_0 = 0.7e-5
        tau_0 = 0.
        args = (X_coor, Y_coor, Z_coor, cam_left, cam_top, traj_2d_left, traj_2d_top, timespan)
        #res_act = least_squares(shift_error, tau_0, args=args)
        res_act = differential_evolution(shift_error, [(-timespan[1], timespan[1])], args=args)
        tau = float(res_act.x)
        X_coor, Y_coor, Z_coor, t1, t2 = get_shifted_3D(tau, X, Y, Z, cam_left, cam_top, traj_2d_left, traj_2d_top, timespan)
        print(tau)

    return X_coor, Y_coor, Z_coor, t1, t2


def shift_error(tau, X, Y, Z, cam_left, cam_top, traj_left, traj_top, timespan):
    """Computes the reprojection error obtained by time shifting the left camera of a value of tau

    :param tau: time shift value
    :param X,Y,Z: unshifted 3D trajectory
    :param cam_left,cam_top: cameras objects
    :param traj_left,traj_top: 2D pixel coordinate trajectory on each camera
    :param
    """
    x, y, z, corr_top, corr_left = get_shifted_3D(tau, X, Y, Z, cam_left, cam_top, traj_left, traj_top, timespan)
    err = [0., 0., 0., 0.]
    len_traj = len(timespan)
    for i in range(0, len_traj-1):
        if not (np.isnan(X[i]) or np.isnan(Y[i]) or np.isnan(Z[i])):
            err_actu = get_proj_error([x[i], y[i], z[i]], cam_left, cam_top, corr_left[i, :], corr_top[i, :])
            w = np.sqrt((x[i + 1] - x[i]) ** 2 + (y[i + 1] - y[i]) ** 2 + (z[i + 1] - z[i]) ** 2)
            if np.sum(np.isnan(err_actu)) == 0 and not(np.isnan(w)):
                err[0] += w * float(err_actu[0]) ** 2
                err[1] += w * float(err_actu[1]) ** 2
                err[2] += w * float(err_actu[2]) ** 2
                err[3] += w * float(err_actu[3]) ** 2

    return np.sum(err)


def get_shifted_3D(tau, X, Y, Z, cam_left, cam_top, traj_left, traj_top, timespan):
    corr_top, corr_left = shift_cam_coord(timespan, traj_top, traj_left, tau)
    len_traj = len(corr_top)
    x = np.zeros(np.shape(X)) * np.nan
    y = np.zeros(np.shape(Y)) * np.nan
    z = np.zeros(np.shape(Z)) * np.nan
    for i in range(0, len_traj):
        if not (np.isnan(X[i]) or np.isnan(Y[i]) or np.isnan(Z[i])):
            A, B = make_system_mat(cam_top, cam_left, corr_left[i, :], corr_top[i, :])
            x[i], y[i], z[i] = np.linalg.solve(np.matrix(A), np.matrix(B).T)

    return x, y, z, corr_top, corr_left


def shift_cam_coord(timespan, traj_2d_top, traj_2d_left, tau=0.):
    """Shifts one of the two cameras of a given time delay to account for asynchronized cameras

    :param timespan: time vector
    :param traj_2d_top: 2D pixel trajectory for the top camera
    :param traj_2d_left: 2D pixel trajectory for the left camera
    :param tau: time_shift
    :return: traj_2d_top, traj_2d_left
    """
    u_interp = interp1d(timespan + tau, traj_2d_left[:, 0])
    v_interp = interp1d(timespan + tau, traj_2d_left[:, 1])
    traj_final = np.nan * np.zeros(traj_2d_left.shape)

    new_time = np.zeros(timespan.shape)
    len_t = len(timespan)
    for i in range(0, len_t):
        if timespan[i] > np.min(timespan + tau) and timespan[i] < np.max(timespan + tau):
            traj_final[i, 0] = u_interp(timespan[i])
            traj_final[i, 1] = v_interp(timespan[i])
    return traj_2d_top, traj_final


def make_system_mat(cam_top, cam_left, pos_2d_left, pos_2d_top):
    """Computes the matrix A, b of the equation system AX = b where X is the shot 3D coordinates

    :param cam_top,cam_left: camera object for the top and left camera
    :param pos_2d_left: position found by the left camera
    :param pos_2d_top: position found by the right camera
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

    A[1, :] = [cam_top.R[2, 0] * v1 - a_top[1, 0], cam_top.R[2, 1] * v1 - a_top[1, 1],
                cam_top.R[2, 2] * v1 - a_top[1, 2]]

    A[2, :] = [cam_left.R[2, 0] * v2 - a_left[1, 0], cam_left.R[2, 1] * v2 - a_left[1, 1],
               cam_left.R[2, 2] * v2 - a_left[1, 2]]

    B[0, 0] = b_top[0, 0] - cam_top.T[2] * u1
    B[0, 1] = b_top[0, 1] - cam_top.T[2] * v1
    B[0, 2] = b_left[0, 1] - cam_left.T[2] * v2
    return A, B


def make_alpha_beta(cam):
    """Compute necessary coefficients for the 3D coordinate system matrix AX=b for better code readability.

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
    """Compute the projection error between the projection of a guessed 3D coordinate vector and the actual screen point

    :param var: 3D coordinate triplet
    :param cam_top,cam_left: camera object for the top and left camera
    :param pos_2d_left: position found by the left camera
    :param pos_2d_top: position found by the right camera
    :return: projection error vector
    """
    X, Y, Z = var
    pos_proj_top = get_proj_coords(X, Y, Z, cam_top)
    pos_proj_left = get_proj_coords(X, Y, Z, cam_left)

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
