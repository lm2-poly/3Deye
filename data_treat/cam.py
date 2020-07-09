import numpy as np
import cv2
import glob
import os

class Cam:
    """
    Class for the camera object
    """
    def __init__(self, mtx, dist, R, T, picDir, firstPic, pic_to_cm, framerate, camRes, res, cropsize=0,
                 origin=(0,0)):
        """
        Camera object initialisation

        :param mtx: camera intrinsinc matrix
        :param dist: camera distortion matrix
        :param R: camera to sample rotation (Rodrigues vector).
        Warning: the initialisation function outputs cam.R as a rotation *matrix* but takes a
        Rodrigues vector as input.
        :param T: camera to sample translation
        :param picDir: shot pictures directory
        :param firstPic: first picture name
        :param pic_to_cm: pixel to cm ratio (deprecated)
        :param framerate: camera framerate
        :param cropsize: size of teh screen to crop (usefull when pictures information was written
        on each pictures)
        :param camRes: camera resolution
        :param res: picture resolution (W, H)
        :param origin: pixel coordinate of the system origin
        """
        self.mtx = np.loadtxt(mtx)
        self.dist = np.loadtxt(dist)
        self.R = np.zeros((3, 3))
        cv2.Rodrigues(np.loadtxt(R), self.R)
        self.T = np.loadtxt(T) #+ np.array([0.75, 0.75, 0])
        self.origin = origin
        self.dir = picDir
        self.firstPic = firstPic
        self.pic_to_cm = pic_to_cm
        self.framerate = framerate
        self.cropSize = cropsize
        self.res = res
        self.camRes = camRes


    def undistort(self):
        """
        Undistort teh camera pictures and change the picture file to the undistorted one
        :return:
        """
        if "corrected" in self.dir.split("/"):
            print("The camera was already calibrated.. exiting")
        else:
            images = glob.glob(self.dir+'/*.jpg')
            if not("corrected" in os.listdir(self.dir)):
                print('mkdir "'+self.dir+'/corrected"')
                os.system('mkdir "'+self.dir+'/corrected"')
            for elem in images:
                img = cv2.imread(elem)
                h, w = img.shape[:2]
                newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, (w, h), 0, (w, h))
                dst = cv2.undistort(img, self.mtx, self.dist, None, newcameramtx)
                dst = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
                cv2.imwrite(self.dir+'/corrected/'+elem.split('\\')[1], dst)

            self.dir = self.dir+"/corrected"