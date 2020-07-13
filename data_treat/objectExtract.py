import glob
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def filter_val(pic, width, height, tol=10., lastVal = [0,0]):
	'''
	Compute the barycenter of a point cloud which pixel grey value is above a given threshold

	:param pic: picture array grey values
	:param width: picture width
	:param height: picture height
	:param tol: filter tolerance (default 100)
	:param lastVal: last barycenter value if several images are treated, used to return something when no values are
	above the threshold
	:return: barycenter x and y coordinates and the number of pixels detected
	'''
	bary_x = 0
	bary_y = 0
	numvals = 0
	for i in range(0, width):
		for j in range(0, height):
			if pic[i,j] > tol:
				bary_x += float(i)
				bary_y += float(j)
				numvals += 1.
	if not(numvals == 0):
		bary_x /= numvals
		bary_y /= numvals
	else:
		bary_x = np.nan
		bary_y = np.nan
	return bary_x, bary_y, numvals


def compute_2d_traj(cam, splitSymb="_", numsplit=1, plotTraj=True):
	'''
	Compute the 2D trajectory (in m) of the barycenter of a moving object filmed by a camera,
	by computing the difference of images with the object and without the object (initial state)

	:param dir: directory name containing the picture
	:param firstPicEnd: name of the first picture (should not contain the moving object)
	:param cropSizes: Size to crop from the picture (bottom), use if the images have a banner
	:param pic_to_cm: pixel to centimeter size ratio
	:param framerate: camera framerate
	:param splitSymb: symbol to use to split the picture names (default "_")
	:param numsplit: place of the image number in the picture name after splitting (default 1)
	:return: X,Y trajectory in the camera reference system and the time list
	'''
	img = Image.open(cam.dir + "/" + cam.firstPic).convert('LA')
	firstNum = int(cam.firstPic.split(splitSymb)[numsplit].split(".")[0])
	width, height = img.size
	area = (cam.cropSize[0], cam.cropSize[2], width - cam.cropSize[1], height - cam.cropSize[3])
	img = img.crop(area)
	RGBPicRef = np.zeros((width, height))
	if (plotTraj):
		imSuper = np.array(img)[:,:,0]
	for i in range(0, width - (cam.cropSize[0] + cam.cropSize[1])):
		for j in range(0, height - (cam.cropSize[2] + cam.cropSize[3])):
			RGBPicRef[i, j] = img.getpixel((i, j))[0]

	img.close()
	picList = glob.glob(cam.dir+"/*.jpg")
	if cam.dir+"/"+cam.firstPic in picList:
		picList.remove(cam.dir+"/"+cam.firstPic)
	else:
		picList.remove(cam.dir+"\\"+cam.firstPic)
	picList = sorted(picList)
	lenDat = len(picList)
	avgdif = np.zeros((lenDat, 2))
	RGBPic_actu = np.zeros((width, height))
	timespan = np.linspace(0, lenDat, lenDat) / cam.framerate
	lastVal = [0., 0.]
	for k in range(0, lenDat):
		img = Image.open(picList[k]).convert('LA')
		img = img.crop(area)
		for i in range(0, width - (cam.cropSize[0] + cam.cropSize[1])):
			for j in range(0, height - (cam.cropSize[2] + cam.cropSize[3])):
				RGBPic_actu[i, j] = img.getpixel((i, j))[0]
		if plotTraj:
			imSuper += np.array(img)[:, :, 0]
			imSuper = (imSuper/2).astype('uint8')
		img.close()
		numActu = int(picList[k].split(splitSymb)[numsplit].split(".")[0]) - firstNum - 1
		bary_x, bary_y, num_pic = filter_val(abs(RGBPic_actu - RGBPicRef), width - (cam.cropSize[0]+cam.cropSize[1]),
											 height - (cam.cropSize[2] + cam.cropSize[3]), lastVal=lastVal)
		#plt.contourf(abs(RGBPic_actu - RGBPicRef))
		#plt.colorbar()		
		#plt.show()
		lastVal = [bary_x, bary_y]
		avgdif[numActu, 0] = bary_x
		avgdif[numActu, 1] = bary_y

		if plotTraj:
			plt.clf()
			plt.imshow(imSuper, cmap='gray')
			plt.plot([bary_x], [bary_y], '.', markersize=5, color="red", label="Detected position")
			plt.xlim((0, cam.res[0]))
			plt.ylim((0, cam.res[1]))
			plt.legend()
			plt.draw()
			plt.pause(0.1)

	#avgdif *= cam.pic_to_cm

	return avgdif, timespan
