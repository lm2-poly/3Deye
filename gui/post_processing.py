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

    entries = makeform(frame, ['Velocity detection factor', 'Output file name'], ["1.3", "Trajectory.txt"])
    b1 = tk.Button(frame, text='Launch Analysis !', command=(lambda x=entries: launch_pp(x, cam_top, cam_left, T, traj_3d)))
    b1.pack(side=tk.TOP, padx=5, pady=5)


def launch_pp(entries, cam_top, cam_left, T, traj_3d):
    plt.close()
    log = ''
    alpha = get_init_angle(traj_3d.X, traj_3d.Y, traj_3d.Z, traj_3d.t, cam_top, cam_left)

    log += 'Angle with horizontal: {:.02f}°\n'.format(alpha)
    xi, yi, zi = get_impact_position(traj_3d.X, traj_3d.Y, traj_3d.Z, cam_left, cam_top)

    log += 'Impact position (cm): ({:.02f}, {:.02f} {:.02f})\n'.format(xi, yi, zi)
    Vinit, Vend = get_velocity(traj_3d.t, traj_3d.X, traj_3d.Y, traj_3d.Z, thres=float(entries[0][1].get()))

    log += 'Initial velocity (m/s): ({:.02f}, {:.02f} {:.02f})\n'.format(Vinit[0]/100., Vinit[1]/100., Vinit[2]/100.)
    log += 'Velocity after impact (m/s): ({:.02f}, {:.02f} {:.02f})\n'.format(Vend[0], Vend[1], Vend[2])
    make_report(traj_3d, alpha, Vinit, Vend, [xi, yi, zi], cam_top, cam_left, entries[1][1].get(),
                "data_treat/report_template.txt")
    log += '\nTrajectory exported as '+entries[1][1].get()
    T.delete('1.0', tk.END)
    T.insert(tk.END, log)
    