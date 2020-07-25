import tkinter as tk
import matplotlib.pyplot as plt
from data_treat.make_report import make_report, load_data
from data_treat.data_pp import get_init_angle, get_impact_position, get_velocity
from gui.gui_utils import makeform, popupmsg


def pp_tab(root,frame, cam_top, cam_left, traj_3d):
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
    pos = makeform(angle_frame, ['Threshold'], ["0.95"])

    vel_frame.pack(side=tk.LEFT)
    angle_frame.pack(side=tk.LEFT)
    pos_frame.pack(side=tk.LEFT)

    params.pack(side=tk.TOP)

    save_frame = tk.Frame(frame)
    file_save = makeform(save_frame, ['Output file name'], ["Trajectory.txt"], pos=tk.TOP)
    b1 = tk.Button(save_frame, text='Save results', command=(lambda fs=file_save:
                                                        save_res(cam_top, cam_left, T, traj_3d, fs)))
    b1.pack(side=tk.BOTTOM, padx=5, pady=5)
    save_frame.pack(side=tk.BOTTOM)


    b1 = tk.Button(frame, text='Launch Analysis !', command=(lambda x=vels, fs=file_save, ang=angles, ps=pos:
                                                             launch_pp(x, cam_top, cam_left, T, traj_3d, fs, ang, ps)))
    b1.pack(side=tk.BOTTOM, padx=5, pady=5)




def save_res(cam_top, cam_left, T, traj_3d, fileSave):
    make_report(traj_3d, cam_top, cam_left, fileSave[0][1].get(),
                "data_treat/report_template.txt")
    log = '\nTrajectory exported as ' + fileSave[0][1].get()
    T.insert(tk.END, log)


def launch_pp(vels, cam_top, cam_left, T, traj_3d, fileSave, ang, pos):
    plt.close()
    log = ''
    alpha = get_init_angle(traj_3d.X, traj_3d.Y, traj_3d.Z, traj_3d.t, cam_top, cam_left,
                           init=int(ang[0][1].get()), end=int(ang[1][1].get()))

    log += 'Angle with horizontal: {:.02f}°\n'.format(alpha)
    xi, yi, zi = get_impact_position(traj_3d.X, traj_3d.Y, traj_3d.Z, cam_left, cam_top, threshold=float(pos[0][1].get()))

    log += 'Impact position (cm): ({:.02f}, {:.02f} {:.02f})\n'.format(xi, yi, zi)
    Vinit, Vend = get_velocity(traj_3d.t, traj_3d.X, traj_3d.Y, traj_3d.Z,
                               thres=float(vels[0][1].get()), init=int(vels[1][1].get()), pt_num=int(vels[2][1].get()))

    log += 'Initial velocity (m/s): ({:.02f}, {:.02f} {:.02f})\n'.format(Vinit[0], Vinit[1], Vinit[2])
    log += 'Velocity after impact (m/s): ({:.02f}, {:.02f} {:.02f})\n'.format(Vend[0], Vend[1], Vend[2])

    traj_3d.set_pp(alpha, Vinit, Vend, [xi, yi, zi])

    T.delete('1.0', tk.END)
    T.insert(tk.END, log)
    