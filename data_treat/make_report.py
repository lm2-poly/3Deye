import numpy as np
from string import Template
from data_treat.cam import Cam
import json


def load_data(fileName, trajectory, cam_top, cam_left):
    """Load existing data from file

    :param fileName: name of the file to load
    """
    fichier = open(fileName)
    str_data = fichier.read()
    fichier.close()

    category = str_data.split("===")
    test_params = category[1].split('\n')
    fps = float(test_params[3].split(':')[1])
    sample = test_params[5].split(':')[1]
    shot = test_params[6].split(':')[1]
    pressure = float(test_params[7].split(':')[1])
    trajectory.set_exp_params(shot, sample, pressure)

    traj = category[3].split('\n')
    t = []
    X = []
    Y = []
    Z = []
    for elem in traj[2:]:
        if (len(elem.split()) == 4):
            ti, xi, yi, zi = elem.split()
        t.append(float(ti))
        X.append(float(xi))
        Y.append(float(yi))
        Z.append(float(zi))
    trajectory.set_trajectory(np.array(t), np.array(X), np.array(Y), np.array(Z))

    cam_top_string = category[5].split("=")[1].split('Top camera\n')[1]
    cam_top.load_from_string(cam_top_string)
    cam_top.framerate = fps
    cam_left_string = category[6].split("=")[1].split('Left camera\n')[1]
    cam_left.load_from_string(cam_left_string)
    cam_left.framerate = fps


def data_save(traj_3d, fileName, cam_top, cam_left):
    """Save position data after 3D trajectory reconstruction

    :param traj_3d: experiment object containing the 3D trajectory
    :return: write the position in a column text file
    """
    lenX = traj_3d.X.shape[0]
    out_str = ''
    for i in range(0, lenX):
        out_str += '\n{:.04E} {:.04E} {:.04E} {:.04E}'.format(traj_3d.t[i], traj_3d.X[i], traj_3d.Y[i], traj_3d.Z[i])
    out_str += "\n\n=== Calibration data\n"
    out_str += "==== Top camera\n" + cam_top.write_cam_data()
    out_str += "==== Left camera\n" + cam_left.write_cam_data()

    fichier = open(fileName, 'a')
    fichier.write(out_str)
    fichier.close()


def make_report(traj_3d, cam_top, cam_left, file_name, template):
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
    make_template_file(template, file_name,
                       [file_name, cam_top.firstPic, cam_top.framerate, cam_top.res, traj_3d.sample,
                       traj_3d.shot, traj_3d.pressure, traj_3d.alpha, traj_3d.vinit[0], traj_3d.vinit[1], traj_3d.vinit[2],
                        traj_3d.vend[0], traj_3d.vend[1], traj_3d.vend[2], traj_3d.impact_pos[0], traj_3d.impact_pos[1], traj_3d.impact_pos[2]],
                       ['testName', 'picName', 'fps', 'res', 'sample', 'shot', 'pressure',
                        'angle', 'VX', 'VY', 'VZ','VXAfter', 'VYAfter', 'VZAfter', 'X', 'Y', 'Z'])
    data_save(traj_3d, file_name, cam_top, cam_left)


def make_template_file(template, sortie, H, var_names):
    """Create a file from a given template file"""
    len_vars = len(H)
    to_rmv = []
    for i in range(0, len_vars):
        if not(type(H[i]) == tuple)  and not(type(H[i]) == str) and np.isnan(H[i]):
            to_rmv.append(var_names[i])
        if type(H[i]) == float or type(H[i]) == np.float64:
            H[i] = '{:.03f}'.format(H[i])

    for elem in to_rmv:
        var_names.remove(elem)
    while 'nan' in H:
        H.remove('nan')

    len_vars = len(H)
    fichier = open(template,'r');
    contenu = fichier.read();
    modif_dic = {}
    for i in range(0, len_vars):
        modif_dic[var_names[i]] = H[i]
    contenu_modif = Template(contenu).safe_substitute(modif_dic);
    fichier.close();
    fichier = open(sortie,'w');
    fichier.write(contenu_modif);
    fichier.close();