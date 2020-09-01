from calibration.calib_len import get_cam_matrix
from calibration.find_sample_ori import get_transfo_mat, plot_proj_origin
from calibration.main import write_calibration_file
import matplotlib.pyplot as plt
import json
import cv2
import numpy as np
from PIL import Image


left_lens = "C:/Users/breum/Desktop/2020-08-26/calib/LEFT"
right_lens = "C:/Users/breum/Desktop/2020-08-26/calib/TOP"
left_pos = "C:/Users/breum/Desktop/2020-08-26/calib_30deg_left.tif"
right_pos = "C:/Users/breum/Desktop/2020-08-26/calib_30deg_top.tif"
calib_folder = "C:/Users/breum/Desktop/2020-08-26/calib_res/30deg"
chess_dim=7
chess_case_len=0.4375

def haris(fname, t1, t2):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dst = cv2.cornerHarris(gray,2,3,t1)
    dst = cv2.dilate(dst,None)
    ret, dst = cv2.threshold(dst,t2*dst.max(),255,0)
    dst = np.uint8(dst)
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)
    plt.imshow(img)
    plt.plot(corners[:,0], corners[:, 1], '.', ms=2, color="red")
    plt.show()
    return corners

corner_top = np.transpose(np.loadtxt("coord_list_top", unpack=True))
corner_left = np.transpose(np.loadtxt("coord_list_left", unpack=True))

print("*** Get top intrinsic matrix")
mtx_top, dist_top = get_cam_matrix(left_lens, chess_dim, chess_case_len)
print("*** Get left intrinsic matrix")
mtx_left, dist_left = get_cam_matrix(right_lens, chess_dim, chess_case_len)

print("*** Get top transformation matrix")
R_top, T_top = get_transfo_mat(left_pos, mtx_top, dist_top, chess_dim, chess_case_len, imgpoints=corner_top)
print("*** Get Left transformation matrix")
R_left, T_left = get_transfo_mat(right_pos, mtx_left, dist_left, chess_dim, chess_case_len, imgpoints=corner_left)



print("Check system coordinate consistency")
plt.figure(figsize=(14,6))
plt.subplot(121)
plt.title("Top camera")
# im = Image.open(right_pos)
plot_proj_origin(right_pos, mtx_top, R_top, T_top, chess_dim, chess_case_len)
# plt.imshow(im, cmap='gray')
# plt.plot(corner_top[:, 0], corner_top[:, 1], '.', color="red")

plt.subplot(122)
plt.title("Left camera")
# im = Image.open(left_pos)
plot_proj_origin(left_pos, mtx_left, R_left, T_left, chess_dim, chess_case_len)
# plt.imshow(im, cmap='gray')
# plt.plot(corner_left[:, 0], corner_left[:, 1], '.', color="red")

plt.show()




print("*** write result file")
write_calibration_file(calib_folder + '/cam_top', mtx_top, dist_top, R_top, T_top)
write_calibration_file(calib_folder + '/cam_left', mtx_left, dist_left, R_left, T_left)
