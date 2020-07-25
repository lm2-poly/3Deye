"""Save position data after 3D trajectory reconstruction"""
import numpy as np
import matplotlib.pyplot as plt
import math
from PIL import Image
import os
from data_treat.reconstruction_3d import get_proj_list
import glob


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

    plt.xlabel('t (ms)')
    plt.ylabel('V (m/s)')
    plt.legend()
    plt.show()


def get_init_angle(Xi, Yi, Zi, ti, cam_top, cam_left, plot=True, saveDir='data_treat/', init=0, end=2):
    """Compute the shot trajectory angle relatively to teh shooting axis
    :param X,Y,Z: reconstructed X, Y, and Z coordinates (ndarray)
    :param timespan: time point list
    :param cam_top,cam_left: camera objects
    :param plot: True or False indicate if should plot a verification picture
    :return: nothing
    """
    numPic = end - init
    t = ti[~np.isnan(Xi)]
    X = Xi[~np.isnan(Xi)]
    Y = Yi[~np.isnan(Xi)]
    Z = Zi[~np.isnan(Xi)]
    dX = X[end] - X[init]
    dY = Y[end] - Y[init]
    dZ = Z[end] - Z[init]

    v = np.array([dX, dY, dZ])/(end-init)

    vnorm = math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
    alpha = math.acos(v[1]/vnorm)*180./math.pi

    xt, yt = get_proj_list(X, Y, Z, cam_top)
    #xl, yl = get_proj_list(-Y, X, Z, cam_left)
    xl, yl = get_proj_list(X, Y, Z, cam_left)

    pos_screen_resize(xt, yt, cam_top)
    pos_screen_resize(xl, yl, cam_left)

    proj_V_t = [xt[end] - xt[init], yt[end] - yt[init]]
    proj_V_l = [xl[end] - xl[init], yl[end] - yl[init]]

    plt.figure(figsize=(14, 6))
    plt.subplot(121)
    plt.title("Left camera")
    plot_supper(init, end, cam_left)
    plt.quiver(xl[init], yl[init],
               proj_V_l[0]/(numPic+1), proj_V_l[1]/(numPic+1), color=(1., 0., 0.), scale=50)
    plt.xlim((0, cam_left.res[0]))
    plt.ylim((0, cam_left.res[1]))

    plt.subplot(122)
    plt.title("Top camera")
    plot_supper(init, end, cam_top)
    plt.quiver([xt[init]], [yt[init]],
               proj_V_t[0]/(numPic+1), proj_V_t[1]/(numPic+1), color=(1., 0., 0.), scale=50)

    plt.xlim((0, cam_top.res[0]))
    plt.ylim((0, cam_top.res[1]))
    if not(saveDir is None):
        plt.savefig(saveDir+"Angle.png")
    if plot:
        plt.show(block=False)
    else:
        plt.close()
    print("Horizontal angle: {:.02f}°".format(alpha))
    return alpha


def get_impact_position(X, Y, Z, cam_left, cam_top, plot=True, saveDir='data_treat/', threshold=0.9):
    """Automatic detection of the moment of impact simply by taking the moment where Y changes direction

    :param X,Y,Z: reconstructed X, Y, and Z coordinates (ndarray)
    :param cam_left,cam_top: left and top camera objects.
    :return: impact X,Y,Z position relative to the first detected shot picture position.
    """
    X0 = X[~np.isnan(X)][0]
    Y0 = Y[~np.isnan(X)][0]
    Z0 = Z[~np.isnan(X)][0]
    i = 0
    lenY = len(Y)
    cont = True
    while i < lenY-1 and cont:
        if Y[i+1] < threshold * Y[i]:
            cont = False
        else:
            i+=1

    plt.figure(figsize=(8, 6))
    plot_supper(0, 10, cam_top)
    xt, yt = get_proj_list(X, Y, Z, cam_top)
    pos_screen_resize(xt, yt, cam_top)
    plt.plot(xt, yt, color="white", label="Shot trajectory")
    plt.plot(xt[i], yt[i], '.', ms=5, color="red", label="Detected impact position")
    plt.title("Impact position detection")
    plt.xlim((0, cam_top.res[0]))
    plt.ylim((0, cam_top.res[1]))
    plt.legend()
    if not(saveDir is None):
        plt.savefig(saveDir+"Impact_position.png")

    if plot:
        plt.show(block=False)
        print("Impact position: ({:.02f}, {:.02f}, {:.02f}) (cm)".format(X[i] - X0, Y[i] - Y0, Z[i] - Z0))
    else:
        plt.close()
    return X[i] - X0, Y[i] - Y0, Z[i] - Z0


def plot_supper(init, end, cam):
    """Plot the superposition (addition) of a cam shot picture between picture init and end
    :param init,end: start and stop indices for the addition
    :param cam: camera object to be used
    """
    picList = glob.glob(cam.dir + "/*.tif")
    picList += glob.glob(cam.dir + "/*.jpg")
    ver_pic = np.array(Image.open(picList[0]))
    ver_pic = ver_pic[0:cam.res[0] - cam.cropSize[3], :]
    for i in range(init, end):
        im_act = np.array(Image.open(picList[i]))
        ver_pic += im_act[0:cam.res[0] - cam.cropSize[3], :]

    plt.imshow(ver_pic, "gray")


def pos_screen_resize(x, y, cam):
    """Returns the coordinate in the resized screen given coordinates in the unresized creen
    :param x,y: ndarray containing unresized scree coordinates
    :param cam: cam object
    :return: nothing but changes x and y values
    """
    x -= (cam.camRes[0] / 2 - cam.res[0] / 2)
    y -= (cam.camRes[1] / 2 - cam.res[1] / 2)


def get_velocity(ti, Xi, Yi, Zi, thres=1.3, plot=True, saveDir='data_treat/', init= 0, pt_num=2):
    """Computes the shot velocity before and after the impact by linear fit.
    Before the impact, the functions continues adding the next acquisition point to the linear fit until the
    new points reduces the fit success score at less then the previous score * threshold. Then the first point with
    constant velocity after the impact is searched so that lienar fit with the same number of points as before the
    impact yields a better score than before the impact. It is based on the assumption that there will always be
    more acquisition point after the impact than before.

    :param t: time vector
    :param X,Y,Z: 3D coordinates (ndarray)
    :param thres: threshold for the accepted residual difference (default 1.3)
    :return: VX,VY,VZ initial velocity vector coordinates
    """
    score_actu = 1000.
    new_score = 100.
    i = init + pt_num
    t = ti[~np.isnan(Xi)]
    X = Xi[~np.isnan(Xi)]
    Y = Yi[~np.isnan(Xi)]
    Z = Zi[~np.isnan(Xi)]
    X0 = X[0]
    Y0 = Y[0]
    Z0 = Z[0]
    len_dat = len(t)

    while i < len_dat and new_score < thres * score_actu:
        score_actu = new_score
        i+=1
        dat = np.polyfit(t[init:i], Y[init:i] - Y0, deg=1, full=True)
        new_score = dat[1][0]

    i-=1
    dat = np.polyfit(t[init:i], Y[init:i] - Y0, deg=1, full=True)
    VX = np.polyfit(t[init:i], X[init:i], deg=1)[0]
    VY = np.polyfit(t[init:i], Y[init:i], deg=1)[0]
    VZ = np.polyfit(t[init:i], Z[init:i], deg=1)[0]

    plt.figure(figsize=(8,6))
    plt.plot(t * 1000, X - X0, marker=".", label="X")
    plt.plot(t * 1000, Y - Y0, marker=".", label="Y")
    plt.plot(t * 1000, Z - Z0, marker=".", label="Z")
    plt.plot(t[init:i] * 1000, (dat[0][0] * t[init:i] + dat[0][1]), label="Best linear fit (initial)")
    print("Initial velocity: ({:.02f}, {:.02f}, {:.02f}) m/s".format(VX / 100, VY / 100, VZ / 100))



    score_down = 1000.
    lenVel = max(i, 3)
    while i + lenVel + 1 < len_dat and score_down > 1.e-4:
        i += 1
        dat = np.polyfit(t[i:i+lenVel], Y[i:i+lenVel] - Y0, deg=1, full=True)
        if dat[0][0] < 0.:
            score_down = float(dat[1])
    dat = np.polyfit(t[i:i+lenVel], Y[i:i+lenVel] - Y0, deg=1, full=True)
    VX_after = np.polyfit(t[i:i + lenVel], X[i:i + lenVel], deg=1)[0]
    VY_after = np.polyfit(t[i:i + lenVel], Y[i:i + lenVel], deg=1)[0]
    VZ_after = np.polyfit(t[i:i + lenVel], Z[i:i + lenVel], deg=1)[0]

    plt.plot(t[i:i+lenVel] * 1000, (dat[0][0] * t[i:i+lenVel] + dat[0][1]), label="Best linear fit (after impact)")
    plt.legend()
    if not(saveDir is None):
        plt.savefig(saveDir+"Velocity.png")
    if plot:
        plt.show(block=False)
        print("Velocity after impact: ({:.02f}, {:.02f}, {:.02f}) m/s".format(VX_after / 100, VY_after / 100,
                                                                      VZ_after / 100))
    else:
        plt.close()
    return [VX / 100, VY / 100, VZ / 100], [VX_after / 100, VY_after / 100, VZ_after / 100]