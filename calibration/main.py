"""
Finds the top and left cameras intrinsinc and distortion matrices and the rotation/translation matrix between the
cameras and the sample
"""
from calibration.calib_len import get_cam_matrix
from calibration.find_sample_ori import get_transfo_mat, plot_proj_origin
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
import json
import cv2


def calibrate_stereo(left_lens, right_lens, left_pos, right_pos, calib_folder, chess_dim,chess_case_len, pic=None):
    """Calibrate a stereocamera system given calibration file names

    :param left_lens,right_lens: path of the right and left camera lens calibration pictures
    :param left_pos,right_pos: path of the right and left camera position pictures
    :param pic: default None, picture fram to print in the gui
    :return: nothing but generates the camera calibration files in the "res folder" mtx_top,mtx_left (camera intrinsinc matrix), dist_top,dist_left (passage matrix from the sample to
    """

    print("Getting cameras matrix")
    mtx_top, dist_top = get_cam_matrix(left_lens, chess_dim, chess_case_len, pic)
    mtx_left, dist_left = get_cam_matrix(right_lens, chess_dim, chess_case_len, pic)

    print("Getting reference frame transformations")
    R_top, T_top = get_transfo_mat(left_pos, mtx_top, dist_top, chess_dim, chess_case_len, pic)
    R_left, T_left = get_transfo_mat(right_pos, mtx_left, dist_left, chess_dim, chess_case_len, pic)
    T_left[0] += chess_case_len * (chess_dim -1 )

    # new_angle = np.linalg.norm(R_top)
    # T_top[0] += chess_case_len * 3 * np.cos(new_angle)
    # T_top[2] += chess_case_len * 3 * np.sin(new_angle)
    # T_top[1] += chess_case_len * 3
    # T_left[0] += chess_case_len * 3
    # T_left[1] += chess_case_len * 3 * np.sin(np.pi/4)
    # T_left[2] -= chess_case_len * 3 * np.cos(np.pi / 4)

    print("Check system coordinate consistency")
    my_dpi = 96
    plt.figure(figsize=(14,6))
    plt.subplot(121)
    plot_proj_origin(left_pos, mtx_top, dist_top, R_top, T_top, "top", chess_dim, chess_case_len)
    plt.title("Top camera")

    plt.subplot(122)
    plt.title("Left camera")
    plot_proj_origin(right_pos, mtx_left, dist_left, R_left, T_left, "left", chess_dim, chess_case_len)

    plt.show(block=False)

    print("Saving results in res")
    write_calibration_file(calib_folder + '/cam_top', mtx_top, dist_top, R_top, T_top)
    write_calibration_file(calib_folder + '/cam_left', mtx_left, dist_left, R_left, T_left)


def write_calibration_file(f_name, mtx, dist, R, T):
    out_str = ''
    out_str += "Intrisinc matrix:\n" + json.dumps(mtx.tolist()) + '\n'
    out_str += "Distorsion matrix:\n" + json.dumps(dist.tolist()) + '\n'
    out_str += "Rotation vector(Rodrigues):\n" + json.dumps(R.tolist()) + '\n'
    out_str += "Translation vector:\n" + json.dumps(T.T.tolist()) + '\n'
    fichier = open(f_name, 'w')
    fichier.write(out_str)
    fichier.close()
