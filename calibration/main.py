"""
Finds the top and left cameras intrinsinc and distortion matrices and the rotation/translation matrix between the
cameras and the sample
"""
from calibration.calib_len import get_cam_matrix
from calibration.find_sample_ori import get_transfo_mat
import numpy as np


def calibrate_stereo(left_lens, right_lens, left_pos, right_pos):
    """Calibrate a stereocamera system given calibration file names

    :param left_lens,right_lens: path of the right and left camera lens calibration pictures
    :param left_pos,right_pos: path of the right and left camera position pictures
    :return: nothing but generates the camera calibration files in the "res folder" mtx_top,mtx_left (camera intrinsinc matrix), dist_top,dist_left (passage matrix from the sample to
    """
    print("Getting cameras matrix")
    mtx_top, dist_top = get_cam_matrix(left_lens)
    mtx_left, dist_left = get_cam_matrix(right_lens)

    print("Getting reference frame transformations")
    R_top, T_top = get_transfo_mat(left_pos, mtx_top, dist_top)
    R_left, T_left = get_transfo_mat(right_pos, mtx_left, dist_left)

    print("Saving results in res")
    np.savetxt("calibration/res/mtx_top", mtx_top)
    np.savetxt("calibration/res/mtx_left", mtx_left)
    np.savetxt("calibration/res/dist_top", dist_top)
    np.savetxt("calibration/res/dist_left", dist_left)
    np.savetxt("calibration/res/R_top", R_top)
    np.savetxt("calibration/res/R_left", R_left)
    np.savetxt("calibration/res/T_top", T_top)
    np.savetxt("calibration/res/T_left", T_left)