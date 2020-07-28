"""
Finds the top and left cameras intrinsinc and distortion matrices and the rotation/translation matrix between the
cameras and the sample.

Most functions were highly inspired from the following source:
https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html

"""
from calibration.calib_len import get_cam_matrix
from calibration.find_sample_ori import get_transfo_mat, plot_proj_origin
import matplotlib.pyplot as plt
import json


def calibrate_stereo(left_lens, right_lens, left_pos, right_pos, calib_folder, chess_dim,chess_case_len):
    """Calibrate a stereocamera system given calibration file names

    :param left_lens,right_lens: path of the right and left camera lens calibration pictures
    :param left_pos,right_pos: path of the right and left camera position pictures
    :param calib_folder: calibration result folder
    :param chess_dim: Number of chess cases -1
    :param chess_case_len: chess case length
    :return: nothing but generates the camera calibration files in the "res folder" mtx_top,mtx_left (camera intrinsinc matrix), dist_top,dist_left (passage matrix from the sample to
    """

    print("Getting cameras matrix")
    mtx_top, dist_top = get_cam_matrix(left_lens, chess_dim, chess_case_len)
    mtx_left, dist_left = get_cam_matrix(right_lens, chess_dim, chess_case_len)

    print("Getting reference frame transformations")
    R_top, T_top = get_transfo_mat(left_pos, mtx_top, dist_top, chess_dim, chess_case_len)
    R_left, T_left = get_transfo_mat(right_pos, mtx_left, dist_left, chess_dim, chess_case_len)

    print("Check system coordinate consistency")
    plt.figure(figsize=(14,6))
    plt.subplot(121)
    plt.title("Top camera")
    plot_proj_origin(left_pos, mtx_top, R_top, T_top, chess_dim, chess_case_len)

    plt.subplot(122)
    plt.title("Left camera")
    plot_proj_origin(right_pos, mtx_left, R_left, T_left, chess_dim, chess_case_len)

    plt.show(block=False)

    print("Saving results in res")
    write_calibration_file(calib_folder + '/cam_top', mtx_top, dist_top, R_top, T_top)
    write_calibration_file(calib_folder + '/cam_left', mtx_left, dist_left, R_left, T_left)


def write_calibration_file(f_name, mtx, dist, R, T):
    """Write a single camera calibration file

    :param f_name: calibration file name to save
    :param mtx: camera intrinsic matrix
    :param dist: camera distorsion matrix
    :param R: Rotation between the camera and the sample coordinate system (Rodrigues vector)
    :param T: Translation vector between the camera and the sample coordinate system
    """
    out_str = ''
    out_str += "Intrisinc matrix:\n" + json.dumps(mtx.tolist()) + '\n'
    out_str += "Distorsion matrix:\n" + json.dumps(dist.tolist()) + '\n'
    out_str += "Rotation vector(Rodrigues):\n" + json.dumps(R.tolist()) + '\n'
    out_str += "Translation vector:\n" + json.dumps(T.T.tolist()) + '\n'
    fichier = open(f_name, 'w')
    fichier.write(out_str)
    fichier.close()
