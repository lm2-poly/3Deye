"""
Reconstruct the 3D trajectory of a shot filmed by two orthogonal cameras
"""
import matplotlib.pyplot as plt
import numpy as np

from reconstruction_3d import reconstruct_3d
from cam import Cam
import math


def make_doublesin_traj(numFrame):
    theta = np.linspace(0., 2 * math.pi, numFrame)
    X = np.linspace(-4, 4., numFrame)
    Y = np.cos(2 * theta)
    Z = np.sin(2 * theta)

    return X, Y, Z


cam_top = Cam("calib_files/mtx_top", "calib_files/dist_top", "calib_files/R_top", "calib_files/T_top",
              "camTop", firstPic="camTop_0000.jpg", pic_to_cm=400 / 500, framerate=20000, cropsize=50)
cam_left = Cam("calib_files/mtx_left", "calib_files/dist_left", "calib_files/R_left", "calib_files/T_left",
               "camLeft", firstPic="camLeft_0000.jpg", pic_to_cm=400 / 500, framerate=20000, cropsize=50)
# un-comment to perform the analyses on undistorted images
#cam_top.make_calib()
#cam_left.make_calib()

X, Y, Z, timespan = reconstruct_3d(cam_top, cam_left, splitSymb="_", numsplit=1)
dt = 1. / cam_top.framerate
plt.figure(figsize=(6, 8))
plt.plot(X, label="X")
plt.plot(Y, label="Y")
plt.plot(Z, label="Z")
plt.legend()
plt.ylim((-2, 2))
plt.show()

