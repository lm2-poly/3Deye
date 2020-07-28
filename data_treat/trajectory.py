import numpy as np


class Experiment:
    """
    Class for the Experiment object
    """
    def __init__(self, t=None, X=None, Y=None, Z=None, shot=None, sample=None, pressure=None, alpha=np.nan,
                 vinit=[np.nan, np.nan, np.nan], vend=[np.nan, np.nan, np.nan], impact_pos=[np.nan, np.nan, np.nan]):
        """Initialization for the Experiment object

        :param t: time list
        :param X,Y,Z: X, Y, Z coordinate lists
        :param shot: shot type (string)
        :param sample: sample name
        :param pressure: pressure in psi (float)
        :param alpha: shot angle wth horizontal
        :param vinit: velocity vector before impact
        :param vend: velocity vector after impact
        :param impact_pos: impact position 3D coordinates
        """
        self.t = t
        self.X = X
        self.Y = Y
        self.Z = Z
        self.shot = shot
        self.sample = sample
        self.pressure = pressure
        self.alpha = alpha
        self.vinit = vinit
        self.vend = vend
        self.impact_pos = impact_pos
        self.save_dir = ""
        self.vel_det_fac = 1.3
        self.vel_init_ind = 1
        self.vel_min_pt = 2
        self.ang_min_ind = 0
        self.ang_end_ind = 3
        self.imp_thres = 0.995

    def set_exp_params(self, shot, sample, pressure, fileName):
        """Set experimental parameters

        :param shot: shot type (string)
        :param sample: sample name
        :param pressure: pressure in psi (float)
        :param fileName: file path to save the experiment report
        """
        self.shot = shot
        self.sample = sample
        self.pressure = pressure
        self.save_dir = fileName

    def set_pp(self, alpha, vinit, vend, impact_pos):
        """Set experiment post-processing values

        :param alpha: shot angle wth horizontal
        :param vinit: velocity vector before impact
        :param vend: velocity vector after impact
        :param impact_pos: impact position 3D coordinates
        """
        self.alpha = alpha
        self.vinit = vinit
        self.vend = vend
        self.impact_pos = impact_pos

    def set_trajectory(self, t, X, Y, Z):
        """Set experiment trajectory list

        :param t: time list
        :param X,Y,Z: X, Y, Z coordinate lists
        """
        self.t = t
        self.X = X
        self.Y = Y
        self.Z = Z