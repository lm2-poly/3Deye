import numpy as np
import cv2
import glob
import os
import json


class Cam:
    """
    Class for the camera object
    """
    def __init__(self, mtx=None, dist=None, R=None, T=None, picDir=None, firstPic=None, pic_to_cm=None,
                 framerate=None, camRes=None, res=None, cropsize=0, origin=(0,0)):
        """Camera object initialisation

        :param mtx: camera intrinsinc matrix
        :param dist: camera distortion matrix
        :param R: camera to sample rotation (Rodrigues vector). Warning: the initialisation function outputs cam.R as a rotation *matrix* but takes a Rodrigues vector as input.
        :param T: camera to sample translation
        :param picDir: shot pictures directory
        :param firstPic: first picture name
        :param pic_to_cm: pixel to cm ratio (deprecated)
        :param framerate: camera framerate
        :param cropsize: size of teh screen to crop (usefull when pictures information was written on each pictures)
        :param camRes: camera resolution
        :param res: picture resolution (W, H)
        :param origin: pixel coordinate of the system origin
        """
        if not(mtx is None):
            self.mtx = np.loadtxt(mtx)
        if not (dist is None):
            self.dist = np.loadtxt(dist)
        self.R = np.zeros((3, 3))
        if not (R is None):
            cv2.Rodrigues(np.loadtxt(R), self.R)
        if not(T is None):
            self.T = np.loadtxt(T) #+ np.array([0.75, 0.75, 0])
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

    def set_mtx(self, mtx):
        self.mtx = np.loadtxt(mtx)

    def set_dist(self, dist):
        self.dist = np.loadtxt(dist)

    def set_R(self, R):
        self.R = np.zeros((3, 3))
        cv2.Rodrigues(np.loadtxt(R), self.R)

    def set_R_by_matrix(self, R):
        self.R = R

    def set_T(self, T):
        self.T = np.loadtxt(T)

    def write_cam_data(self):
        out_str = ''
        out_str += "Screen resolution:\n" + json.dumps(self.camRes) + '\n'
        out_str += "Acquisition resolution:\n" + json.dumps(self.res) + '\n'
        out_str += "Crop array:\n" + json.dumps(self.cropSize) + '\n'
        out_str += "Intrisinc matrix:\n" + json.dumps(self.mtx.tolist())+'\n'
        out_str += "Distorsion matrix:\n" + json.dumps(self.dist.tolist())+'\n'
        out_str += "Rotation matrix:\n" + json.dumps(self.R.tolist()) + '\n'
        out_str += "Translation vector:\n" + json.dumps(self.T.tolist()) + '\n'
        out_str += "Picture directory: \n"+self.dir + '\n'
        out_str += "First picture: \n"+str(self.firstPic) + '\n'
        return out_str

    def load_from_string(self, data):
        lines = data.split('\n')
        self.camRes = tuple(json.loads(lines[1]))
        self.res = tuple(json.loads(lines[3]))
        self.cropSize = json.loads(lines[5])
        self.mtx = np.matrix(json.loads(lines[7]))
        self.dist = np.matrix(json.loads(lines[9]))
        self.R = np.matrix(json.loads(lines[11]))
        self.T = np.matrix(json.loads(lines[13]))
        self.dir = lines[15]
        self.firstPic = int(lines[17])