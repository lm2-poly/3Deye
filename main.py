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
#calibrate_stereo("calibration/lens_dist_calib_top", "calibration/lens_dist_calib_left",
#                "calibration/sources/calib_checker_top.jpg", "calibration/sources/calib_checker_left.jpg")
calib_file = "calibration/res"

print("******* Loading camera data and props")
cam_top = Cam(calib_file+"/mtx_top", calib_file+"/dist_top",
              calib_file+"/R_top", calib_file+"/T_top",
              "camTop", firstPic="150psi1_0001.jpg", pic_to_cm=1 / 141.1, framerate=31000,
              res=(336, 288), cropsize=[0, 0, 0, 50], origin=(187, 658)) #[0, 50, 0, 145]
cam_left = Cam(calib_file+"/mtx_left", calib_file+"/dist_left",
               calib_file+"/R_left", calib_file+"/T_left",
               "camLeft", firstPic="150psi1_0001.jpg", pic_to_cm=1 / 148.97, framerate=31000,
               res=(336, 288), cropsize=[0, 0, 0, 50], origin=(412, 916)) #[207, 0, 0, 50]

#print("******* Removing lens distorsion")
#cam_top.undistort()
#cam_left.undistort()

print("******* Reconstructing 3D trajectory")
X, Y, Z, timespan = reconstruct_3d(cam_top, cam_left, splitSymb="_", numsplit=1, method="persp")
X_ana, Y_ana, Z_ana = make_doublesin_traj(20)

print("******* Plot results")
plt.figure(figsize=(12, 8))
plt.subplot(121)
plt.plot(timespan[2:]*1000, X[~np.isnan(X)], marker=".", label="X")
plt.plot(timespan[2:]*1000, Y[~np.isnan(X)], marker=".", label="Y")
plt.plot(timespan[2:]*1000, Z[~np.isnan(X)], marker=".", label="Z")
plt.xlabel('t (ms)')
plt.ylabel('Z (cm)')
plt.subplot(122)
dt = 1./cam_left.framerate
plt.plot(timespan[3:]*1000, np.diff(X[~np.isnan(X)])/dt/100, marker=".", label="$v_X$")
plt.plot(timespan[3:]*1000, np.diff(Y[~np.isnan(X)])/dt/100, marker=".", label="$v_Y$")
plt.plot(timespan[3:]*1000, np.diff(Z[~np.isnan(X)])/dt/100, marker=".", label="$v_Z$")
x_0 = timespan[3]*1000
y_0 = np.diff(X[~np.isnan(X)])[0]/np.diff(timespan)[0]/100.
plt.annotate("$v_0$: {:.02f}m/s".format(-y_0), (x_0, y_0))
plt.xlabel('t (ms)')
plt.ylabel('V (m/s)')
plt.legend()
#plt.xlim((0,10))
#plt.ylim((-2.0, 2.0))
plt.show()

print("******* Export trajectory file")
data_save(timespan, X, Y, Z)
