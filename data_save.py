"""Save position data after 3D trajectory reconstruction"""
import numpy as np


def data_save(t, X, Y, Z):
    """Save position data after 3D trajectory reconstruction

    :param t: time list
    :param X,Y,Z: position lists
    :return: write the position in a column text file
    """
    np.savetxt("Trajectory.txt", np.array([np.matrix(t).T, np.matrix(X).T,
                                           np.matrix(Y).T, np.matrix(Z).T]).T[0])
