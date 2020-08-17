"""Several post processing functions to provide the shot velocity before and after impact, the impact angle and position..."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
from PIL import Image
from data_treat.reconstruction_3d import get_proj_list
import glob
from gui.gui_utils import plot_fig
from matplotlib.figure import Figure


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
    """Compute the shot trajectory angle relatively to the shooting axis

    :param X,Y,Z: reconstructed X, Y, and Z coordinates (ndarray)
    :param timespan: time point list
    :param cam_top,cam_left: camera objects
    :param plot: True or False indicate if should plot a verification picture
    :param saveDir: Directory to save the picture to
    :param init,end: initial and final array index used to average the angle value (default: 0 and 2)
    """
    numPic = end - init
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
    xl, yl = get_proj_list(X, Y, Z, cam_left)

    pos_screen_resize(xt, yt, cam_top)
    pos_screen_resize(xl, yl, cam_left)

    proj_V_t = [xt[end] - xt[init], yt[end] - yt[init]]
    proj_V_l = [xl[end] - xl[init], yl[end] - yl[init]]

    #f = plt.figure(figsize=(8, 6))
    f = Figure(figsize=(8, 6))
    ax1, ax2 = f.subplots(ncols=2, nrows=1)

    ax1.set_title("Left camera")
    plot_supper(init, end, cam_left, ax=ax1)
    projNorm = np.sqrt(proj_V_l[0]**2 + proj_V_l[1]**2)
    ax1.quiver(xl[init], yl[init],
               10.*proj_V_l[0]/projNorm, 10.*proj_V_l[1]/projNorm, color=(1., 0., 0.), scale=50)
    ax1.set_xlim((0, cam_left.res[0]))
    ax1.set_ylim((0, cam_left.res[1]))

    ax2.set_title("Top camera")
    projNorm = np.sqrt(proj_V_l[0] ** 2 + proj_V_l[1] ** 2)
    plot_supper(init, end, cam_top, ax=ax2)
    ax2.quiver([xt[init]], [yt[init]],
              10.*proj_V_t[0]/projNorm, 10.*proj_V_t[1]/projNorm, color=(1., 0., 0.), scale=50)

    ax2.set_xlim((0, cam_top.res[0]))
    ax2.set_ylim((0, cam_top.res[1]))

    if not(saveDir is None):
        f.savefig(saveDir+"Angle.png")
    if plot:
        #plt.show(block=False)
        plot_fig(f)
        print("Horizontal angle: {:.02f}°".format(alpha))
    # else:
    #     plt.close(f)

    return alpha


def get_impact_position(X, Y, Z, cam_left, cam_top, plot=True, saveDir='data_treat/', threshold=0.995):
    """Automatic detection of the moment of impact simply by taking the moment where Y changes direction, within a given threshold

    :param X,Y,Z: reconstructed X, Y, and Z coordinates (ndarray)
    :param cam_left,cam_top: left and top camera objects.
    :param plot: True or False indicate if should plot a verification picture
    :param saveDir: Directory to save the picture to
    :param threshold: impact detection threshold
    :return: impact X,Y,Z position relative to the first detected shot picture position.
    """
    i = 0
    lenY = len(Y)
    cont = True
    while i < lenY-1 and cont:
        if Y[i+1] < threshold * Y[i]:
            cont = False
        else:
            i+=1

    f = Figure(figsize=(8, 6))
    ax = f.subplots(ncols=1, nrows=1)
    plot_supper(0, 10, cam_top, ax=ax)
    xt, yt = get_proj_list(X, Y, Z, cam_top)
    pos_screen_resize(xt, yt, cam_top)
    ax.plot(xt, yt, color="white", label="Shot trajectory")
    ax.plot(xt[i], yt[i], '.', ms=5, color="red", label="Detected impact position")
    ax.set_title("Impact position detection")
    ax.set_xlim((0, cam_top.res[0]))
    ax.set_ylim((0, cam_top.res[1]))
    ax.legend()
    if not(saveDir is None):
        f.savefig(saveDir+"Impact_position.png")

    if plot:
        #plt.show(block=False)
        plot_fig(f)
        print("Impact position: ({:.02f}, {:.02f}, {:.02f}) (cm)".format(X[i], Y[i], Z[i]))
    # else:
    #     plt.close(f)
    return X[i], Y[i], Z[i]


def plot_supper(init, end, cam, thres=40., ax=None):
    """Plot the superposition (addition) of a cam shot picture between picture init and end

    :param init,end: start and stop indices for the addition
    :param cam: camera object to be used
    """
    picList = glob.glob(cam.dir + "/*.tif")
    picList += glob.glob(cam.dir + "/*.jpg")
    pic_init = np.array(Image.open(picList[0]))
    ver_pic = pic_init[0:cam.res[0] - cam.cropSize[3], :]
    for i in range(init, end):
        im_act = np.array(Image.open(picList[i]))

        mask = np.abs(im_act - pic_init)
        mask = mask[0:cam.res[0] - cam.cropSize[3], :]
        mask[mask < thres] = 0
        mask[mask >= thres] = 1

        ver_pic = ver_pic * (1 - mask) + im_act[0:cam.res[0] - cam.cropSize[3], :] * mask
    ver_pic = ver_pic / (end-init)
    ver_pic = ver_pic.astype('uint8')
    if ax is None:
        plt.imshow(ver_pic, "gray")
    else:
        ax.imshow(ver_pic, "gray")


def pos_screen_resize(x, y, cam):
    """Returns the coordinate in the resized screen given the coordinates in the usnresized creen (when cropping for higher fps)

    :param x,y: ndarray containing unresized scree coordinates
    :param cam: cam object
    :return: nothing but changes x and y values
    """
    x -= (cam.camRes[0] / 2 - cam.res[0] / 2)
    y -= (cam.camRes[1] / 2 - cam.res[1] / 2)


def get_velocity(ti, Xi, Yi, Zi, thres=1.3, plot=True, saveDir='data_treat/', init=0, pt_num=2):
    """Computes the shot velocity before and after the impact by linear fit.
    Before the impact, the functions continues adding the next acquisition point to the linear fit until the
    new points reduces the fit success score at less then the previous score * threshold. Then the first point with
    constant velocity after the impact is searched so that lienar fit with the same number of points as before the
    impact yields a better score than before the impact. It is based on the assumption that there will always be
    more acquisition point after the impact than before.

    :param t: time vector
    :param Xi,Yi,Zi: 3D coordinates (ndarray)
    :param thres: threshold for the accepted residual difference (default 1.3)
    :param plot: True or False indicate if should plot a verification picture
    :param saveDir: Directory to save the picture to
    :param init: initial index to compute the initial velocity
    :param pt_num: minimum number of points to use to compute the velocity
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

    f = Figure(figsize=(8, 6))
    ax = f.subplots(ncols=1, nrows=1)
    ax.plot(t * 1000, X - X0, marker=".", label="X")
    ax.plot(t * 1000, Y - Y0, marker=".", label="Y")
    ax.plot(t * 1000, Z - Z0, marker=".", label="Z")
    ax.plot(t[init:i] * 1000, (dat[0][0] * t[init:i] + dat[0][1]), label="Best linear fit (initial)")
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

    ax.plot(t[i:i+lenVel] * 1000, (dat[0][0] * t[i:i+lenVel] + dat[0][1]), label="Best linear fit (after impact)")
    ax.legend()
    if not(saveDir is None):
        f.savefig(saveDir+"Velocity.png")
    if plot:
        #plt.show(block=False)
        plot_fig(f)
        print("Velocity after impact: ({:.02f}, {:.02f}, {:.02f}) m/s".format(VX_after / 100, VY_after / 100,
                                                                      VZ_after / 100))
    # else:
    #     plt.close(f)
    return [VX / 100, VY / 100, VZ / 100], [VX_after / 100, VY_after / 100, VZ_after / 100]