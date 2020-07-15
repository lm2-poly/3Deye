"""
Reconstruct the 3D trajectory of a shot filmed by two orthogonal cameras
"""
import matplotlib.pyplot as plt
import numpy as np

from data_treat.reconstruction_3d import reconstruct_3d, cam_shift_origin
from data_treat.cam import Cam
from calibration.main import calibrate_stereo
from data_treat.data_pp import result_plot, get_init_angle, get_impact_position, get_velocity
from data_treat.make_report import make_report, load_data

# print("******* Calibrating cameras")
# calibrate_stereo("calibration/lens_dist_calib_top",
#                  "calibration/lens_dist_calib_left",
#                  "calibration/sources/calib_checker_top.jpg",
#                  "calibration/sources/calib_checker_left.jpg")


print("******* Loading camera data and props")
calib_file = "calibration/res"  # Camera calibration file folder
cam_top = Cam(calib_file + "/mtx_top",  # Top Camera intrinsic matrix file
              calib_file + "/dist_top",  # Top Camera distorsion parameters file
              calib_file + "/R_top",  # Top Camera Rotation matrix file
              calib_file + "/T_top",  # Top Camera Translation vector file
              "camTop",  # Top Camera picture folder
              firstPic="camTop_0000.jpg",  # Top Camera first picture name
              pic_to_cm=1 / 141.1,  # Top Camera pixel to cm conversion ratio
              framerate=15000,  # Top Camera framerate
              camRes=(500, 500),  # Camera resolution
              res=(500, 500),  # Resized picture resolution (after cropping the metadata banner)
              cropsize=[0, 0, 0, 0],  # Image crop size (X_start, X_end, Y_start, Y_end)
              origin=(187, -50))  # origin=(187, 658)) #Top Camera frame origin
# [0, 50, 0, 145]

cam_left = Cam(calib_file + "/mtx_left",
               calib_file + "/dist_left",
               calib_file + "/R_left",
               calib_file + "/T_left",
               "camLeft",
               firstPic="camLeft_0000.jpg",
               pic_to_cm=1 / 148.97,
               framerate=15000,
               camRes=(500, 500),
               res=(500, 500),
               cropsize=[0, 0, 0, 0],
               origin=(412, 916))
# [207, 0, 0, 50]
#cam_shift_origin(cam_left)
#cam_shift_origin(cam_top)

# print("******* Removing lens distorsion")
# cam_top.undistort()
# cam_left.undistort()


print("******* Reconstructing 3D trajectory")
X, Y, Z, timespan = reconstruct_3d(cam_top, cam_left, splitSymb="_", numsplit=-1, method="persp", plotTraj=False)
#timespan, X, Y, Z = load_data("TestBlender.txt")

print("******* Plot results")
result_plot(X, Y, Z, timespan)

print("******* Compute initial angle")
alpha = get_init_angle(X, Y, Z, timespan, cam_top, cam_left)

print("******* Compute impact location")
xi, yi, zi = get_impact_position(X, Y, Z, cam_left, cam_top)

print("******* Compute Velocity values")
Vinit, Vend = get_velocity(timespan, X, Y, Z)

print("******* Export trajectory file and report")
make_report(timespan, X, Y, Z, alpha, Vinit, Vend, [xi, yi, zi], cam_top, cam_left, "TestBlender")
