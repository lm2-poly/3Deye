import numpy as np


class Experiment:
    """
    Class for the Trajectory object
    """
    def __init__(self, t=None, X=None, Y=None, Z=None, shot=None, sample=None, pressure=None, alpha=np.nan,
                 vinit=[np.nan, np.nan, np.nan], vend=[np.nan, np.nan, np.nan], impact_pos=[np.nan, np.nan, np.nan]):
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

    def set_exp_params(self, shot, sample, pressure, fileName):
        self.shot = shot
        self.sample = sample
        self.pressure = pressure
        self.save_dir = fileName

    def set_pp(self, alpha, vinit, vend, impact_pos):
        self.alpha = alpha
        self.vinit = vinit
        self.vend = vend
        self.impact_pos = impact_pos

    def set_trajectory(self, t, X, Y, Z):
        self.t = t
        self.X = X
        self.Y = Y
        self.Z = Z