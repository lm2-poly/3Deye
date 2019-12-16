"""
Reconstruct the 3D trajectory of a shot filmed by two orthogonal cameras
"""
import matplotlib.pyplot as plt
import numpy as np

from reconstruction_3d import reconstruct_3d
from cam import Cam


def make_doublesin_traj(numFrame):
    X = np.linspace(-4, 4., numFrame)
    Y = np.linspace(-2., 2., numFrame)
    Z = -np.linspace(-2., 2., numFrame)

    return X, Y, Z


cam_top = Cam("calib_files/mtx_top", "calib_files/dist_top", "calib_files/R_top", "calib_files/T_top",
              "camTop", firstPic="camTop_0000.jpg", pic_to_cm=1 / 139.5, framerate=20000,
              res=(500, 500), cropsize=50)
cam_left = Cam("calib_files/mtx_left", "calib_files/dist_left", "calib_files/R_left", "calib_files/T_left",
               "camLeft", firstPic="camLeft_0000.jpg", pic_to_cm=1 / 139.5, framerate=20000,
               res=(500, 500), cropsize=50)
# un-comment to perform the analyses on undistorted images
#cam_top.undistort()
#cam_left.undistort()

X, Y, Z, timespan = reconstruct_3d(cam_top, cam_left, splitSymb="_", numsplit=1, method="persp")
X_ana, Y_ana, Z_ana = make_doublesin_traj(20)

plt.figure(figsize=(6, 8))
plt.plot(X[~np.isnan(X)], marker=".", label="X")
plt.plot(Y[~np.isnan(X)], marker=".", label="Y")
plt.plot(Z[~np.isnan(X)], marker=".", label="Z")
plt.plot(X_ana[~np.isnan(X)][1:], label="X initial")
plt.plot(Y_ana[~np.isnan(X)][1:], label="Y initial")
plt.plot(Z_ana[~np.isnan(X)][1:], label="Z initial")
plt.legend()
plt.xlim((0,10))
plt.ylim((-2.0, 2.0))
plt.show()

