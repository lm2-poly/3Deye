import tkinter as tk
import matplotlib.pyplot as plt
from data_treat.make_report import make_report
from data_treat.data_pp import get_init_angle, get_impact_position, get_velocity
from gui.gui_utils import makeform, popupmsg
import os
import platform
import numpy as np


def pp_tab(root,frame, cam_top, cam_left, traj_3d):
    """Setup the post-processing tab

    :param root: tk root window
    :param frame: calibration tab frame object
    :param cam_top,cam_left: top and left camera objects
    :param traj_3d: Experiment object
    """
    pic = tk.Frame(frame, width=500, height=350,
                   highlightbackground="black", highlightthickness=1.)
    T = tk.Text(pic)
    T.insert(tk.END, '')
    T.pack()
    pic.pack(side=tk.TOP)
    params = tk.Frame(frame)
    vel_frame = tk.Frame(params)
    title = tk.Label(vel_frame, text="Velocity determination parameters")
    title.pack(side=tk.TOP)
    vels = makeform(vel_frame, ['Velocity detection factor', 'Initial index', 'Minimum number of points'], ["1.3", '1', '2'])

    angle_frame = tk.Frame(params)
    title = tk.Label(angle_frame, text="Angle determination parameters")
    title.pack(side=tk.TOP)
    angles = makeform(angle_frame, ['Initial index', 'End index'], ["0", "3"])

    pos_frame = tk.Frame(params)
    title = tk.Label(angle_frame, text="Impact position parameters")
    title.pack(side=tk.TOP)
    pos = makeform(angle_frame, ['Threshold'], ["0.995"])

    vel_frame.pack(side=tk.LEFT)
    angle_frame.pack(side=tk.LEFT)
    pos_frame.pack(side=tk.LEFT)

    params.pack(side=tk.TOP)

    save_frame = tk.Frame(frame)
    file_save = makeform(save_frame, ['Output file name'], [traj_3d.save_dir], pos=tk.TOP)
    b1 = tk.Button(save_frame, text='Save results', command=(lambda fs=file_save:
                                                        save_res(cam_top, cam_left, T, traj_3d, fs)))
    b1.pack(side=tk.BOTTOM, padx=5, pady=5)
    save_frame.pack(side=tk.BOTTOM)


    b1 = tk.Button(frame, text='Launch Analysis !', command=(lambda x=vels, fs=file_save, ang=angles, ps=pos:
                                                             launch_pp(x, cam_top, cam_left, T, traj_3d, fs, ang, ps)))
    b1.pack(side=tk.BOTTOM, padx=5, pady=5)


def save_res(cam_top, cam_left, T, traj_3d, fileSave):
    """Saves the results and make a report

    :param cam_top,cam_left: top and left camera objects
    :param T: Text canvas to log the results
    :param traj_3d: Experiment object
    :param fileSave: report save directory
    """
    list_pic = os.listdir('data_treat')
    try:
        traj_3d.save_dir = fileSave[0][1].get()
        if 'Angle.png' in list_pic:
            move_file('data_treat/Angle.png', parse_dir(traj_3d.save_dir))
        if 'Impact_position.png' in list_pic:
            move_file('data_treat/Impact_position.png', parse_dir(traj_3d.save_dir))
        if 'Velocity.png' in list_pic:
            move_file('data_treat/Velocity.png', parse_dir(traj_3d.save_dir))
        if 'Reproj_error.png' in list_pic:
            move_file('data_treat/Reproj_error.png', parse_dir(traj_3d.save_dir))

        make_report(traj_3d, cam_top, cam_left, "data_treat/report_template.txt")
        log = '\nTrajectory exported as ' + traj_3d.save_dir
        T.insert(tk.END, log)

    except:
        popupmsg("Wrong save directory")
    else:
        popupmsg("File saved successfully !")


def parse_dir(save_dir):
    """Parses the save directory to get only the directory (without the file name) to save the pictures to

    :param save_dir: save directory string
    :return: save directory string without the file name
    """
    dir_list = save_dir.split('/')
    dir = ''
    if len(dir_list) == 1:
        dir_list = dir_list[0].split('\\')
    if not(len(dir_list) == 1):
        for elem in dir_list[:-1]:
            dir += elem + '/'

    return dir


def move_file(init, end):
    """Moves a file using either batch or shell command according to the current os (Windows or Linux only)

    :param init: path of the file to move
    :param end: target directory path
    """
    if platform.system() == 'Windows':
        init = init.replace('/', '\\')
        end = end.replace('/', '\\')
        os.system('copy '+init+' '+end)
    elif platform.system() == 'Linux':
        os.system('scp -r '+init+' '+end)
    else:
        popupmsg("Unknown OS, I cannot save the pictures. You may find them in the data_treat folder directly...")


def launch_pp(vels, cam_top, cam_left, T, traj_3d, fileSave, ang, pos):
    """Launch a post-processing analysis

    :param vels: velocity parameter form
    :param cam_top,cam_left: top and left camera objects
    :param T: Text canvas to log the results
    :param traj_3d: Experiment object
    :param fileSave: report save directory
    :param ang: angle parameter form
    :param pos: impact position parameter form
    """
    #plt.close()
    log = ''
    fileSave[0][1].delete(0, tk.END)
    fileSave[0][1].insert(tk.END, traj_3d.save_dir)

    alpha = get_init_angle(traj_3d.X, traj_3d.Y, traj_3d.Z, traj_3d.t, cam_top, cam_left,
                           init=int(ang[0][1].get()), end=int(ang[1][1].get()))

    log += 'Angle with horizontal: {:.02f}°\n'.format(alpha)
    xi, yi, zi = get_impact_position(traj_3d.X, traj_3d.Y, traj_3d.Z, cam_left, cam_top, threshold=float(pos[0][1].get()))

    log += 'Impact position (cm): ({:.02f}, {:.02f} {:.02f})\n'.format(xi, yi, zi)
    Vinit, Vend = get_velocity(traj_3d.t, traj_3d.X, traj_3d.Y, traj_3d.Z,
                               thres=float(vels[0][1].get()), init=int(vels[1][1].get()), pt_num=int(vels[2][1].get()))

    vi_norm = np.sqrt(Vinit[0] ** 2 + Vinit[1] ** 2 + Vinit[2] ** 2)
    vo_norm = np.sqrt(Vend[0] ** 2 + Vend[1] ** 2 + Vend[2] ** 2)

    log += 'Initial velocity (m/s): ({:.02f}, {:.02f} {:.02f})\n'.format(Vinit[0], Vinit[1], Vinit[2])
    log += 'Initial velocity norm (m/s): {:.02f}\n'.format(vi_norm)
    log += 'Velocity after impact (m/s): ({:.02f}, {:.02f} {:.02f})\n'.format(Vend[0], Vend[1], Vend[2])
    log += 'Velocity norm after impact (m/s): {:.02f}\n'.format(vo_norm)

    traj_3d.set_pp(alpha, Vinit, Vend, [xi, yi, zi])

    T.delete('1.0', tk.END)
    T.insert(tk.END, log)
