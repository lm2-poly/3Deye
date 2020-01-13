"""
Reconstruct the 3D trajectory of a shot filmed by two orthogonal cameras
"""
import matplotlib.pyplot as plt
import numpy as np

from reconstruction_3d import reconstruct_3d
from cam import Cam
from calibration.main import calibrate_stereo
from data_save import data_save

def make_doublesin_traj(numFrame):
    X = np.linspace(-4, 4., numFrame)
    Y = np.linspace(-2., 2., numFrame)
    Z = -np.linspace(-2., 2., numFrame)

    return X, Y, Z


#print("******* Calibrating cameras")
#calibrate_stereo("calibration/lens_dist_calib", "calibration/lens_dist_calib",
#                "calibration/sources/tilted_top.png", "calibration/sources/tilted_left.png")
calib_file = "calibration/res"

print("******* Loading camera data and props")
cam_top = Cam(calib_file+"/mtx_top", calib_file+"/dist_top",
              calib_file+"/R_top", calib_file+"/T_top",
              "camTop", firstPic="camTop_0000.jpg", pic_to_cm=1 / 139.5, framerate=20000,
              res=(500, 500), cropsize=50)
cam_left = Cam(calib_file+"/mtx_left", calib_file+"/dist_left",
               calib_file+"/R_left", calib_file+"/T_left",
               "camLeft", firstPic="camLeft_0000.jpg", pic_to_cm=1 / 139.5, framerate=20000,
               res=(500, 500), cropsize=50)

# print("******* Removing lens distorsion")
#cam_top.undistort()
#cam_left.undistort()

print("******* Reconstructing 3D trajectory")
X, Y, Z, timespan = reconstruct_3d(cam_top, cam_left, splitSymb="_", numsplit=1, method="persp")
X_ana, Y_ana, Z_ana = make_doublesin_traj(20)

print("******* Plot results")
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

print("******* Export trajectory file")
data_save(timespan, X, Y, Z)
