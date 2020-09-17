import glob
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from gui.gui_utils import plot_fig
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
import time


def create_circle(center, radius):
	theta = np.linspace(0., 2 * np.pi, 100)
	x = center[0] + radius * np.cos(theta)
	y = center[1] + radius * np.sin(theta)
	return x, y


def filter_val(pic, width, height, tol=20.):
	"""Compute the barycenter of a point cloud which pixel grey value is above a given threshold

	:param pic: picture array grey values
	:param width: picture width
	:param height: picture height
	:param tol: filter tolerance (default 20)
	:return: barycenter x and y coordinates and the number of pixels detected
	"""
	xi = np.arange(width)
	yi = np.arange(height)
	Y, X = np.meshgrid(yi, xi)
	bool_tab = pic>tol

	# mask = (bool_tab * 1).astype('uint8')
	# cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	# cnts = imutils.grab_contours(cnts)
	# center = None
	# if len(cnts) > 0:
	# 	c = max(cnts, key=cv2.contourArea)
	# 	((x, y), radius) = cv2.minEnclosingCircle(c)
	# else:
	# 	x = np.nan
	# 	y = np.nan
	# 	radius = np.nan

	bary_x = float(np.sum(X[bool_tab]))
	bary_y = float(np.sum(Y[bool_tab]))
	numvals = np.sum(bool_tab)
	
	if not(numvals == 0):
		bary_x /= numvals
		bary_y /= numvals
	else:
		bary_x = np.nan
		bary_y = np.nan
	return bary_x, bary_y, numvals


def compute_2d_traj(cam, splitSymb="_", numsplit=-1, plotTraj=True, isgui=False):
	"""Compute the 2D trajectory (in m) of the barycenter of a moving object filmed by a camera,
	by computing the difference of images with the object and without the object (initial state)

	:param cam: camera object
	:param splitSymb: symbol to use to split the picture names (default "_")
	:param numsplit: place of the image number in the picture name after splitting (default -1)
	:param plotTraj: TRue or False, indicate if the detected point should be plotted
	:return: X,Y trajectory in the camera reference system and the time list
	"""
	picList = glob.glob(cam.dir + "/*.tif")
	picList += glob.glob(cam.dir + "/*.jpg")
	picList = sorted(picList)

	firstNum = picList[0].split(splitSymb)[numsplit].split(".")[0]
	if int(firstNum) == 0:
		num0 = len(firstNum)-1
	else:
		num0 = (firstNum.count('0') - int(np.floor(np.log10(cam.firstPic))))

	firstPic_name = picList[0].split(num0*'0')[0] + num0 * '0'+str(cam.firstPic)+ '.' + picList[0].split('.')[1]
	img = Image.open(firstPic_name).convert('LA')
	firstNum = cam.firstPic

	width, height = img.size
	RGBPicRef = (np.array(img)[:, :, 0].T).astype(np.int16)
	RGBPicRef[:int(cam.mask_w), :] = 0 #Width mask
	RGBPicRef[:, RGBPicRef.shape[1] - int(cam.mask_h):] = 0 #Height mask
	RGBPicRef = RGBPicRef[:, :height - cam.cropSize[3]] #Vertical crop
	RGBPicRef = RGBPicRef[cam.cropSize[0]:width - cam.cropSize[1], :] #Horizontal crop

	if (plotTraj):
		imSuper = np.copy(RGBPicRef.T)
	img.close()

	if firstPic_name in picList:
		picList.remove(firstPic_name)
	else:
		picList.remove(firstPic_name)

	lenDat = len(picList)
	avgdif = np.zeros((lenDat, 2))
	timespan = np.linspace(0, lenDat, lenDat) / cam.framerate

	if isgui and plotTraj:
		fig = Figure(figsize=(8, 6))
		root, canvas = plot_fig(fig)

	for k in range(0, lenDat):
		img = Image.open(picList[k]).convert('LA')
		RGBPic_actu = (np.array(img)[:, :, 0].T).astype(np.int16)
		RGBPic_actu[:int(cam.mask_w), :] = 0
		RGBPic_actu[:, RGBPic_actu.shape[1] - int(cam.mask_h):] = 0
		RGBPic_actu = RGBPic_actu[:, :height - cam.cropSize[3]]
		RGBPic_actu = RGBPic_actu[cam.cropSize[0]:width - cam.cropSize[1], :]
		if plotTraj:
			imSuper = np.copy(RGBPic_actu.T)
		img.close()

		numActu = int(picList[k].split(splitSymb)[numsplit].split(".")[0]) - firstNum - 1
		bary_x, bary_y, num_pic = filter_val(abs(RGBPic_actu - RGBPicRef), width - (cam.cropSize[0]+cam.cropSize[1]),
											 height - (cam.cropSize[2] + cam.cropSize[3]), tol=cam.cam_thres)

		avgdif[numActu, 0] = bary_x
		avgdif[numActu, 1] = bary_y

		if plotTraj:
			if not(isgui):
				plt.clf()
				plt.imshow(imSuper, cmap='gray')
				plt.plot([bary_x], [bary_y], '.', markersize=3, color="red", label="Detected position")
			else:
				fig.clear()
				fig.add_subplot(111).imshow(imSuper, cmap='gray')
				fig.add_subplot(111).plot(bary_x, bary_y, '.', ms=2, color='red')
				# if not(np.isnan(circ_radius)):
				# 	x_c, y_c = create_circle((bary_x, bary_y), circ_radius)
				# 	canvas.figure.add_subplot(111).plot(x_c, y_c, color='red', linewidth=1)
				# 	canvas.figure.add_subplot(111).plot(bary_x, bary_y, '.', ms=2, color='red')

			# plt.xlim((0, cam.res[0]))
			# plt.ylim((0, cam.res[1]))
			# plt.legend()

			if canvas is None:
				plt.draw()
				plt.pause(0.1)
			else:
				canvas.draw()
				time.sleep(0.1)
	if isgui and plotTraj:
		root.destroy()

	return avgdif, timespan
