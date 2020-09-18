import numpy as np
from string import Template
from gui.gui_utils import popupmsg
import glob
import platform
import os


def load_data(fileName, trajectory, cam_top, cam_left):
    """Load existing data from file

    :param fileName: name of the file to load
    :param trajectory: Experiment object to initialize
    :param cam_top, cam_left: top and left camera object to initialize
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
    trajectory.set_exp_params(shot, sample, pressure, fileName)

    traj = category[3].split('\n')
    t = []
    X = []
    Y = []
    Z = []
    yerr = []
    for elem in traj[2:]:
        if (len(elem.split()) > 4):
            if (len(elem.split()) == 4):
                ti, xi, yi, zi = elem.split()
            elif (len(elem.split()) == 5):
                ti, xi, yi, zi, yerri = elem.split()
                yerr.append(float(yerri))
            t.append(float(ti))
            X.append(float(xi))
            Y.append(float(yi))
            Z.append(float(zi))

    if len(yerr) == 0:
        trajectory.set_trajectory(np.array(t), np.array(X), np.array(Y), np.array(Z))
    else:
        trajectory.set_trajectory(np.array(t), np.array(X), np.array(Y), np.array(Z), yerr=np.array(yerr))

    cam_top_string = category[5].split("=")[1].split('Top camera\n')[1]
    cam_top.load_from_string(cam_top_string)
    cam_top.framerate = fps
    cam_left_string = category[6].split("=")[1].split('Left camera\n')[1]
    cam_left.load_from_string(cam_left_string)
    cam_left.framerate = fps

    listpic = glob.glob("data_treat/*.png")
    for elem in listpic:
        if "Reproj_error.png" in elem:
            delete_pic("data_treat/Reproj_error.png")


def delete_pic(pic_path):
    if platform.system() == 'Windows':
        pic_path = pic_path.replace('/', '\\')
        os.system('del '+pic_path)
    elif platform.system() == 'Linux':
        os.system('rm -rf '+pic_path)
    else:
        popupmsg("Unknown OS, I cannot save the pictures. You may find them in the data_treat folder directly...")


def data_save(traj_3d, cam_top, cam_left):
    """Save experiment data after 3D trajectory reconstruction

    :param traj_3d: experiment object to save
    :param cam_top, cam_left: top and left camera object to save
    :return: 1 if success
    """

    lenX = traj_3d.X.shape[0]
    out_str = ''
    for i in range(0, lenX):
        out_str += '\n{:.04E} {:.04E} {:.04E} {:.04E}'.format(traj_3d.t[i], traj_3d.X[i],
                                                              traj_3d.Y[i], traj_3d.Z[i])
        if not(traj_3d.yerr is None):
            out_str += ' {:.04E}'.format(traj_3d.yerr[i])

    out_str += "\n\n=== Calibration data\n"
    out_str += "==== Top camera\n" + cam_top.write_cam_data()
    out_str += "==== Left camera\n" + cam_left.write_cam_data()

    fichier = open(traj_3d.save_dir, 'a')
    fichier.write(out_str)
    fichier.close()
    return 1


def make_report(traj_3d, cam_top, cam_left, template):
    """Generates a report of the post-processed values as well as the parameters used to extract the trajectory.
    Also saves the trajectory.

    :param traj_3d: Experiment object
    :param cam_top,cam_left: camera objects used for the trajectory determination
    :param template: name of the template to use for the report
    """
    if traj_3d.save_dir == '':
        popupmsg('Empty file name')
        return 0
    vi_norm = np.sqrt(traj_3d.vinit[0] ** 2 + traj_3d.vinit[1] ** 2 + traj_3d.vinit[2] ** 2)
    vo_norm = np.sqrt(traj_3d.vend[0] ** 2 + traj_3d.vend[1] ** 2 + traj_3d.vend[2] ** 2)
    dimp = traj_3d.yerr[traj_3d.impact_i]
    make_template_file(template, traj_3d.save_dir,
                       [traj_3d.save_dir, cam_top.firstPic, cam_top.framerate, cam_top.res, traj_3d.sample,
                       traj_3d.shot, traj_3d.pressure, traj_3d.alpha, traj_3d.vinit[0], traj_3d.vinit[1], traj_3d.vinit[2],
                        vi_norm, traj_3d.vend[0], traj_3d.vend[1], traj_3d.vend[2], vo_norm,
                        traj_3d.impact_pos[0], traj_3d.impact_pos[1], traj_3d.impact_pos[2], dimp],
                       ['testName', 'picName', 'fps', 'res', 'sample', 'shot', 'pressure',
                        'angle', 'VX', 'VY', 'VZ', 'ViNorm', 'VXAfter', 'VYAfter', 'VZAfter', 'VoNorm',
                        'X', 'Y', 'Z', 'dimp'])
    data_save(traj_3d, cam_top, cam_left)


def make_template_file(template, sortie, H, var_names):
    """Create a file from a given template file

    :param template: template file name
    :param sortie: output file name
    :param H: variable values
    :param var_names: variable names
    """
    len_vars = len(H)
    to_rmv = []
    for i in range(0, len_vars):
        if type(H[i]) == float or type(H[i]) == np.float64:
            H[i] = '{:.04f}'.format(H[i])

    for elem in to_rmv:
        var_names.remove(elem)

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