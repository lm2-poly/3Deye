"""Save position data after 3D trajectory reconstruction"""
import numpy as np
import matplotlib.pyplot as plt
import math
from PIL import Image
import os
from data_treat.reconstruction_3d import get_proj_list


def load_data(fileName):
    """Load existing data from file

    :param fileName: name of the file to load
    """
    t, X, Y, Z = np.loadtxt(fileName, unpack=True)
    return t, X, Y, Z


def data_save(t, X, Y, Z):
    """Save position data after 3D trajectory reconstruction

    :param t: time list
    :param X,Y,Z: position lists
    :return: write the position in a column text file
    """
    np.savetxt("Trajectory.txt", np.array([np.matrix(t).T, np.matrix(X).T,
                                           np.matrix(Y).T, np.matrix(Z).T]).T[0])


def result_plot(X, Y, Z, timespan):
    """Plot the recovered shot trajectory and velocity

    :param X,Y,Z: reconstructed X, Y, and Z coordinates (ndarray)
    :param timespan: time point list
    :return: nothing
    """
    X0 = X[~np.isnan(X)][0]
    Y0 = Y[~np.isnan(X)][0]
    Z0 = Z[~np.isnan(X)][0]
    plt.figure(figsize=(12, 8))
    plt.subplot(121)
    plt.plot(timespan[~np.isnan(X)] * 1000, X[~np.isnan(X)] - X0, marker=".", label="X")
    plt.plot(timespan[~np.isnan(X)] * 1000, Y[~np.isnan(X)] - Y0, marker=".", label="Y")
    plt.plot(timespan[~np.isnan(X)] * 1000, Z[~np.isnan(X)] - Z0, marker=".", label="Z")
    plt.xlabel('t (ms)')
    plt.ylabel('Z (cm)')
    plt.subplot(122)
    dt = timespan[1] - timespan[0]
    plt.plot(timespan[~np.isnan(X)][1:] * 1000, np.diff(X[~np.isnan(X)]) / dt / 100, marker=".", label="$v_X$")
    plt.plot(timespan[~np.isnan(X)][1:] * 1000, np.diff(Y[~np.isnan(X)]) / dt / 100, marker=".", label="$v_Y$")
    plt.plot(timespan[~np.isnan(X)][1:] * 1000, np.diff(Z[~np.isnan(X)]) / dt / 100, marker=".", label="$v_Z$")
    x_0 = timespan[3] * 1000
    y_0 = np.diff(X[~np.isnan(X)])[0] / np.diff(timespan)[0] / 100.
    plt.annotate("$v_0$: {:.02f}m/s".format(-y_0), (x_0, y_0))
    plt.xlabel('t (ms)')
    plt.ylabel('V (m/s)')
    plt.legend()
    plt.show()


def get_init_angle(X, Y, Z, t, cam_top, cam_left, plot=True):
    """Compute the shot trajectory angle relatively to teh shooting axis
    :param X,Y,Z: reconstructed X, Y, and Z coordinates (ndarray)
    :param timespan: time point list
    :param cam_top,cam_left: camera objects
    :param plot: True or False indicate if should plot a verification picture
    :return: nothing
    """
    init = 1
    end = 6
    numPic = end - init
    dX = X[end] - X[init] #np.diff(X[~np.isnan(X)], n=2)
    dY = Y[end] - Y[init] #np.diff(Y[~np.isnan(X)], n=2)
    dZ = Z[end] - Z[init] #np.diff(Z[~np.isnan(X)], n=2)

    v = np.array([dX, dY, dZ])/(end-init)

    vnorm = math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
    alpha = math.acos(v[1]/vnorm)*180./math.pi

    xt, yt = get_proj_list(X, Y, Z, cam_top)
    xl, yl = get_proj_list(-Y, Z, -X, cam_left)

    pos_screen_resize(xt, yt, cam_top)
    pos_screen_resize(xl, yl, cam_left)

    proj_V_t = [xt[end] - xt[init], yt[end] - yt[init]]
    proj_V_l = [xl[end] - xl[init], yl[end] - yl[init]]

    plt.figure(figsize=(14, 6))
    plt.subplot(121)
    plt.title("Left camera")
    plot_supper(init, end, cam_left)
    plt.quiver(xl[init] - cam_left.origin[0], yl[init] - cam_left.origin[1],
               proj_V_l[0]/(numPic+1), proj_V_l[1]/(numPic+1), color=(1., 0., 0.), scale=50)
    plt.xlim((0, cam_left.res[0]))
    plt.ylim((0, cam_left.res[1]))

    plt.subplot(122)
    plt.title("Top camera")
    plot_supper(init, end, cam_top)
    plt.quiver([xt[init]- cam_top.origin[0]], [yt[init]- cam_top.origin[1]],
               proj_V_t[0]/(numPic+1), proj_V_t[1]/(numPic+1), color=(1., 0., 0.), scale=50)

    plt.xlim((0, cam_top.res[0]))
    plt.ylim((0, cam_top.res[1]))
    plt.show()

    print("Horizontal angle: {:.02f}°".format(alpha))
    return alpha


def plot_supper(init, end, cam):
    picList = os.listdir(cam.dir)
    ver_pic = np.array(Image.open(cam.dir + '/' + picList[0]))
    ver_pic = ver_pic[0:cam.res[0] - cam.cropSize[3], :]
    for i in range(init, end):
        im_act = np.array(Image.open(cam.dir + '/' + picList[i]))
        ver_pic += im_act[0:cam.res[0] - cam.cropSize[3], :]

    plt.imshow(ver_pic, "gray")



def pos_screen_resize(x, y, cam):
    x -= (cam.camRes[0] / 2 - cam.res[0] / 2)
    y -= (cam.camRes[1] / 2 - cam.res[1] / 2)
