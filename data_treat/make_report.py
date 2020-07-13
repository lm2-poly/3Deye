import numpy as np
from string import Template

def load_data(fileName):
    """Load existing data from file

    :param fileName: name of the file to load
    """
    t, X, Y, Z = np.loadtxt(fileName, unpack=True)
    return t, X, Y, Z


def data_save(t, X, Y, Z, fileName):
    """Save position data after 3D trajectory reconstruction

    :param t: time list
    :param X,Y,Z: position lists
    :return: write the position in a column text file
    """
    np.savetxt(fileName+".txt", np.array([np.matrix(t).T, np.matrix(X).T,
                                           np.matrix(Y).T, np.matrix(Z).T]).T[0])


def make_report(t, X, Y, Z, alpha, Vinit, Vend, imp_pos, cam_top, cam_left, file_name):
    """Generates a report of the post-processed values as wel as the parameters used to extract the trajectory.
    Also saves the trajectory.
    :param t: timespan vector
    :param X,Y,Z: estimated 3D trajectory nd arrays
    :param alpha: Shot horizontal angle
    :param Vinit,Vend: shot velocity before and after the impact
    :param imp_pos: x,y,z coordinate list of the impact position
    :param cam_top,cam_left: camera objects used for the trajectory determination
    :param file_name: name of the trajectory file generated
    """
    data_save(t, X, Y, Z, file_name)
    make_template_file("data_treat/report_template.txt", "Report.txt",
                       [file_name, cam_top.firstPic, cam_top.framerate, cam_top.res, alpha, Vinit[0]/100, Vinit[1]/100, Vinit[2]/100,
                        Vend[0]/100, Vend[1]/100, Vend[2]/100, imp_pos[0], imp_pos[1], imp_pos[2]],
                       ['testName', 'picName', 'fps', 'res', 'angle', 'VX', 'VY', 'VZ',
                        'VXAfter', 'VYAfter', 'VZAfter', 'X', 'Y', 'Z'])


def make_template_file(template, sortie, H, var_names):
    """Create a file from a given template file"""
    len_vars = len(H)
    for i in range(0, len_vars):
        if type(H[i]) == float or type(H[i]) == np.float64:
            H[i] = '{:.03f}'.format(H[i])
    fichier = open(template,'r');
    contenu = fichier.read();
    modif_dic = {}
    for i in range(0, len_vars):
        modif_dic[var_names[i]] = H[i]
    contenu_modif = Template(contenu).substitute(modif_dic);
    fichier.close();
    fichier = open(sortie,'w');
    fichier.write(contenu_modif);
    fichier.close();