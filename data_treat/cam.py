import numpy as np
import cv2
import glob
import os
import json
from PIL import Image


class Cam:
    """
    Class for the camera object
    """
    def __init__(self, calib_file=None, picDir=None, firstPic=None, pic_to_cm=None,
                 framerate=None, camRes=None, res=None, cropsize=0, cam_thres=20.):
        """Camera object initialisation

        :param calib_file: calibration file for the camera
        :param picDir: shot pictures directory
        :param firstPic: first picture index
        :param pic_to_cm: pixel to cm ratio
        :param framerate: camera framerate
        :param camRes: camera resolution
        :param res: picture resolution (W, H). Warning, it is the picture resolution written on each picture e.g. not accounting for the banner.
        :param cropsize: size of the screen to crop (usefull when pictures information was written on each pictures)
        :param cam_thres: shot detection trheshold
        """
        if not(calib_file is None):
            self.load_calibration_file(calib_file)
        self.dir = picDir
        self.firstPic = firstPic
        self.pic_to_cm = pic_to_cm
        self.framerate = framerate
        self.cropSize = cropsize
        self.res = res
        self.camRes = camRes
        if not(res is None) and not(camRes is None):
            self.set_crop_size()
        self.cam_thres = cam_thres
        self.mask_w = 0
        self.mask_h = 0

    def undistort(self):
        """Undistort the camera pictures and change the picture file to the undistorted one"""

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

    def set_mask(self, mask_w, mask_h):
        """Set black mask to apply on each picture to remove reflexive surfaces

        :param mask_w, mask_h: mask width and height in pixels
        """
        self.mask_w = mask_w
        self.mask_h = mask_h

    def set_mtx(self, mtx):
        """Set camera intrinsic matrix for a given path name (deprecated)

        :param mtx: camera intrinsic matrix file name
        """
        self.mtx = np.loadtxt(mtx)

    def set_dist(self, dist):
        """Set camera distorsion matrix for a given path name (deprecated)

        :param dist: camera distorsion matrix file name
        """
        self.dist = np.loadtxt(dist)

    def set_R(self, R):
        """Set camera rotation matrix for a given Rodrigues vector

        :param R: camera rotation Rodrigues vector
        """
        self.R = np.zeros((3, 3))
        cv2.Rodrigues(R, self.R)

    def set_R_by_matrix(self, R):
        """Set camera rotation matrix for a given matrix

        :param R: camera rotation matrix
        """
        self.R = R

    def set_T(self, T):
        """Set camera Translation vector

        :param T: camera Translation vector
        """
        self.T = np.loadtxt(T)

    def write_cam_data(self):
        """Writes all camera data into a formated tring

        :return: String containing all relevant camera data
        """
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

    def load_calibration_file(self, f_name):
        """Load camera intrinsic, distorsion and transformation matrices from a calibration file

        :param f_name:calibration file path
        """
        fichier = open(f_name)
        lines = fichier.read().split('\n')
        fichier.close()

        self.mtx = np.matrix(json.loads(lines[1]))
        self.dist = np.matrix(json.loads(lines[3]))
        R = np.matrix(json.loads(lines[5]))
        self.set_R(R)
        self.T = np.array(json.loads(lines[7])[0])

    def load_from_string(self, data):
        """Initialize a camera object from a formatted string such as produced by write_cam_data

        :param data: formated string to parse
        """
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

    def set_crop_size(self):
        """Set camera crop size to remove the banner according to the picture effective resolution and the picture target resolution (cam.res)"""
        picList = glob.glob(self.dir + "/*.tif")
        picList += glob.glob(self.dir + "/*.jpg")
        testpic = np.array(Image.open(picList[0]))
        hor_crop = int(np.abs(testpic.shape[1] - self.res[0])/2.)
        self.cropSize = [hor_crop, hor_crop, 0, testpic.shape[0] - self.res[1]]
