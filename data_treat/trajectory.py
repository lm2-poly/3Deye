import numpy as np


class Experiment:
    """
    Class for the Trajectory object
    """
    def __init__(self, t=None, X=None, Y=None, Z=None, shot=None, sample=None, pressure=None):
        self.t = t
        self.X = X
        self.Y = Y
        self.Z = Z
        self.shot = shot
        self.sample = sample
        self.pressure = pressure

    def set_exp_params(self, shot, sample, pressure):
        self.shot = shot
        self.sample = sample
        self.pressure = pressure

    def set_trajectory(self, t, X, Y, Z):
        self.t = t
        self.X = X
        self.Y = Y
        self.Z = Z