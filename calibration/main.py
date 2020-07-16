"""
Finds the top and left cameras intrinsinc and distortion matrices and the rotation/translation matrix between the
cameras and the sample
"""
from calibration.calib_len import get_cam_matrix
from calibration.find_sample_ori import get_transfo_mat, plot_proj_origin
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def calibrate_stereo(left_lens, right_lens, left_pos, right_pos, pic=None):
    """Calibrate a stereocamera system given calibration file names

    :param left_lens,right_lens: path of the right and left camera lens calibration pictures
    :param left_pos,right_pos: path of the right and left camera position pictures
    :param pic: default None, picture fram to print in the gui
    :return: nothing but generates the camera calibration files in the "res folder" mtx_top,mtx_left (camera intrinsinc matrix), dist_top,dist_left (passage matrix from the sample to
    """
    print("Getting cameras matrix")
    mtx_top, dist_top = get_cam_matrix(left_lens, pic)
    mtx_left, dist_left = get_cam_matrix(right_lens, pic)

    print("Getting reference frame transformations")
    R_top, T_top = get_transfo_mat(left_pos, mtx_top, dist_top, pic)
    R_left, T_left = get_transfo_mat(right_pos, mtx_left, dist_left, pic)
    T_left[0] += 0.4375 * 6

    print("Check system coordinate consistency")
    my_dpi = 96
    figure1 = plt.figure(figsize=(300/my_dpi, 300/my_dpi), dpi=my_dpi)
    ax1 = figure1.add_subplot(111)
    if not(pic is None):
        bar1 = FigureCanvasTkAgg(figure1, pic)
        bar1.get_tk_widget().pack(side=tk.LEFT)
    plot_proj_origin(left_pos, mtx_top, dist_top, R_top, T_top)
    plt.title("Top camera")

    figure2 = plt.figure(figsize=(300 / my_dpi, 300 / my_dpi), dpi=my_dpi)
    ax2 = figure2.add_subplot(111)
    if not (pic is None):
        bar2 = FigureCanvasTkAgg(figure2, pic)
        bar2.get_tk_widget().pack(side=tk.LEFT)
    plt.title("Left camera")
    plot_proj_origin(right_pos, mtx_left, dist_left, R_left, T_left)

    if not (pic is None):
        plt.show()

    print("Saving results in res")
    np.savetxt("calibration/res/mtx_top", mtx_top)
    np.savetxt("calibration/res/mtx_left", mtx_left)
    np.savetxt("calibration/res/dist_top", dist_top)
    np.savetxt("calibration/res/dist_left", dist_left)
    np.savetxt("calibration/res/R_top", R_top)
    np.savetxt("calibration/res/R_left", R_left)
    np.savetxt("calibration/res/T_top", T_top)
    np.savetxt("calibration/res/T_left", T_left)