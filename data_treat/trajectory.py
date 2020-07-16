import numpy as np


class Trajectory:
    """
    Class for the Trajectory object
    """
    def __init__(self, t=None, X=None, Y=None, Z=None):
        self.t = t
        self.X = X
        self.Y = Y
        self.Z = Z

    def set_trajectory(self, t, X, Y, Z):
        self.t = t
        self.X = X
        self.Y = Y
        self.Z = Z