"""
Reconstruct the 3D trajectory of a shot filmed by two orthogonal cameras
"""
import matplotlib.pyplot as plt
import numpy as np

from data_treat.reconstruction_3d import reconstruct_3d
from data_treat.cam import Cam
from calibration.main import calibrate_stereo
from data_treat.data_pp import result_plot, get_init_angle, get_impact_position, get_velocity
from data_treat.make_report import make_report, load_data
from gui.recons_gui import start_gui
from data_treat.trajectory import Experiment


print("******* Calibrating cameras")
# calibrate_stereo("calibration/lens_dist_calib_top",
#                  "calibration/lens_dist_calib_left",
#                  "calibration/sources/calib_checker_top.jpg",
#                  "calibration/sources/calib_checker_left.jpg",
#                  "calibration/res/",
#                  chess_dim=7,
#                  chess_case_len=0.4375)


print("******* Loading camera data and props")
calib_file = "calibration/res/cam_top"  # Camera calibration file folder
cam_top = Cam(calib_file,
              "tests\\single\\camTop",  # Top Camera picture folder
              firstPic=0,  # Top Camera first picture name
              pic_to_cm=1 / 141.1,  # Top Camera pixel to cm conversion ratio
              framerate=15000,  # Top Camera framerate
              camRes=(1280, 800),  # Camera resolution
              res=(1280, 800))  # Resized picture resolution (after cropping the metadata banner)

calib_file = "calibration/res/cam_left"  # Camera calibration file folder
cam_left = Cam(calib_file,
               "tests\\single\\camLeft",
               firstPic=0,
               pic_to_cm=1 / 148.97,
               framerate=15000,
               camRes=(1280, 800),
               res=(1280, 800))

# print("******* Removing lens distorsion")
# cam_top.undistort()
# cam_left.undistort()


print("******* Reconstructing 3D trajectory")
X, Y, Z, timespan, err = reconstruct_3d(cam_top, cam_left, splitSymb="_", numsplit=-1, method="persp", plotTraj=False)
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
traj_3d = Experiment()
traj_3d.set_exp_params("0.5mm", "Al6051", "100", "Trajectory.txt")
traj_3d.set_trajectory(timespan, X, Y, Z)
traj_3d.set_pp(alpha, Vinit, Vend, [xi, yi, zi])
make_report(traj_3d, cam_top, cam_left, "data_treat\\report_template.txt")
