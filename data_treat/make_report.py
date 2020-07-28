import numpy as np
from string import Template
from gui.gui_utils import popupmsg


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


def data_save(traj_3d, cam_top, cam_left):
    """Save experiment data after 3D trajectory reconstruction

    :param traj_3d: experiment object to save
    :param cam_top, cam_left: top and left camera object to save
    :return: 1 if success
    """

    lenX = traj_3d.X.shape[0]
    out_str = ''
    for i in range(0, lenX):
        out_str += '\n{:.04E} {:.04E} {:.04E} {:.04E}'.format(traj_3d.t[i], traj_3d.X[i], traj_3d.Y[i], traj_3d.Z[i])
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

    make_template_file(template, traj_3d.save_dir,
                       [traj_3d.save_dir, cam_top.firstPic, cam_top.framerate, cam_top.res, traj_3d.sample,
                       traj_3d.shot, traj_3d.pressure, traj_3d.alpha, traj_3d.vinit[0], traj_3d.vinit[1], traj_3d.vinit[2],
                        traj_3d.vend[0], traj_3d.vend[1], traj_3d.vend[2], traj_3d.impact_pos[0], traj_3d.impact_pos[1], traj_3d.impact_pos[2]],
                       ['testName', 'picName', 'fps', 'res', 'sample', 'shot', 'pressure',
                        'angle', 'VX', 'VY', 'VZ','VXAfter', 'VYAfter', 'VZAfter', 'X', 'Y', 'Z'])
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
            H[i] = '{:.03f}'.format(H[i])

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