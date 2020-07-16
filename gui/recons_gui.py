import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from calibration.main import calibrate_stereo
import matplotlib.pyplot as plt
from data_treat.cam import Cam
from data_treat.trajectory import Trajectory
from data_treat.reconstruction_3d import reconstruct_3d
from data_treat.make_report import make_report, load_data
from data_treat.data_pp import get_init_angle, get_impact_position, get_velocity
import numpy as np
import json


def start_gui():
    cam_top = Cam()
    cam_left = Cam()
    traj_3d = Trajectory()

    root = tk.Tk()
    root.title("Eye3D")
    root.geometry("1000x600")
    style = ttk.Style(root)
    style.configure("lefttab.TNotebook", tabposition="wn")

    notebook = ttk.Notebook(root, style="lefttab.TNotebook")

    calib_frame = tk.Frame(notebook, width=800, height=600)
    ana_frame = tk.Frame(notebook, width=800, height=600)
    pp_frame = tk.Frame(notebook, width=800, height=600)
    root.update_idletasks()

    calib_tab(root, calib_frame)
    ana_tab(root, ana_frame, notebook, cam_top, cam_left, traj_3d)
    pp_tab(root, pp_frame, cam_top, cam_left, traj_3d)

    notebook.add(calib_frame, text="Calibration")
    notebook.add(ana_frame, text="Analysis")
    notebook.add(pp_frame, text="Post processing", state="disabled")

    notebook.grid(row=0, column=0, sticky="n")

    root.mainloop()


def calib_tab(root,frame):
    pic = tk.Frame(frame, width=500, height=350,
                   highlightbackground="black", highlightthickness=1.)

    ents = makeform(frame, ['Top camera chessboard picture folder',
                            'Left camera chessboard picture folder',
                            'Top camera sample position picture',
                            'Left camera sample position picture'],
                    ["calibration/lens_dist_calib_top",
                  "calibration/lens_dist_calib_left",
                  "calibration/sources/calib_checker_top.jpg",
                  "calibration/sources/calib_checker_left.jpg"])

    #frame.bind('<Return>', (lambda event, e=ents: launch_calib(e, pic)))
    b1 = tk.Button(frame, text='Calibrate !',
                   command=(lambda e=ents: launch_calib(e, pic)))
    pic.pack(side=tk.TOP)
    b1.pack(side=tk.TOP, padx=5, pady=5)


def ana_tab(root,frame, notebook, cam_top, cam_left, traj_3d):
    top_cam = tk.Frame(frame, width=250)
    titleTop = tk.Label(top_cam, text="Top camera parameters")
    titleTop.pack(side=tk.TOP)
    top = makeform(top_cam, ['Calibration folder',"Picture folder", 'First picture name', 'framerate',
                             'Screen width', 'Screen height', "Acquisition width", "Acquisition height",
                             "Banner crop size"],
                    ["calibration/res", "camTop", "camTop_0000.jpg", '15000',
                     "500", "500", "500", "500", "0"])

    left_cam = tk.Frame(frame, width=250)
    titleLeft = tk.Label(left_cam, text="Left camera parameters")
    titleLeft.pack(side=tk.TOP)
    left = makeform(left_cam, ['Calibration folder',"Picture folder", 'First picture name', 'framerate',
                             'Screen width', 'Screen height', "Acquisition width", "Acquisition height",
                             "Banner crop size"],
                    ["calibration/res", "camLeft", "camLeft_0000.jpg", '15000',
                     "500", "500", "500", "500", "0"])
    b1 = tk.Button(frame, text='Launch Analysis !',
                   command=(lambda t=top, l=left: launch_analysis(t, l, notebook, w.get(), cam_top, cam_left, traj_3d)))
    b1.pack(side=tk.TOP, padx=5, pady=5)
    w = ttk.Combobox(frame, values=['No perspective', 'Perspective simple', 'Perspective optimized'])
    w.insert(tk.END, 'Perspective simple')
    w.pack(side=tk.TOP)
    top_cam.pack(side=tk.LEFT, padx=5, pady=5)
    left_cam.pack(side=tk.RIGHT, padx=5, pady=5)


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
    log = ''
    alpha = get_init_angle(traj_3d.X, traj_3d.Y, traj_3d.Z, traj_3d.t, cam_top, cam_left)

    log += 'Angle with horizontal: {:.02f}°\n'.format(alpha)
    xi, yi, zi = get_impact_position(traj_3d.X, traj_3d.Y, traj_3d.Z, cam_left, cam_top)

    log += 'Impact position (cm): ({:.02f}, {:.02f} {:.02f})\n'.format(xi, yi, zi)
    Vinit, Vend = get_velocity(traj_3d.t, traj_3d.X, traj_3d.Y, traj_3d.Z, thres=float(entries[0][1].get()))

    log += 'Initial velocity (m/s): ({:.02f}, {:.02f} {:.02f})\n'.format(Vinit[0]/100., Vinit[1]/100., Vinit[2]/100.)
    log += 'Velocity after impact (m/s): ({:.02f}, {:.02f} {:.02f})\n'.format(Vend[0], Vend[1], Vend[2])
    make_report(traj_3d.t, traj_3d.X, traj_3d.Y, traj_3d.Z, alpha, Vinit, Vend, [xi, yi, zi], cam_top, cam_left, entries[1][1].get(),
                "data_treat/report_template.txt")
    log += 'Report exported as Report.txt\nTrajectory exported as '+entries[1][1].get()
    T.delete('1.0', tk.END)
    T.insert(tk.END, log)


def launch_calib(entries, pic):
    calibrate_stereo(entries[0][1].get(),
                     entries[1][1].get(),
                     entries[2][1].get(),
                     entries[3][1].get(), pic)


def create_camera(entries, name, cam):
    cam.set_mtx(entries[0][1].get() + "/mtx_"+name)
    cam.set_dist(entries[0][1].get() + "/dist_"+name)
    cam.set_R(entries[0][1].get() + "/R_"+name)
    cam.set_T(entries[0][1].get() + "/T_"+name)
    cam.dir = entries[1][1].get()
    cam.firstPic = entries[2][1].get()
    cam.pic_to_cm = 1 / 141.1
    cam.framerate = float(entries[3][1].get())
    cam.camRes = (int(entries[4][1].get()), int(entries[5][1].get()))
    cam.res = (int(entries[6][1].get()), int(entries[7][1].get()))
    cam.cropSize = [0, 0, 0, int(entries[8][1].get())]
    return cam


def launch_analysis(top_entry, left_entry, notebook, method, cam_top, cam_left, traj_3d):
    create_camera(top_entry, 'top', cam_top)
    create_camera(left_entry, 'left', cam_left)
    notebook.tab(2, state='normal')
    X, Y, Z, timespan = reconstruct_3d(cam_top, cam_left,
                                       splitSymb="_", numsplit=-1, method=method, plotTraj=False)
    traj_3d.set_trajectory(timespan, X, Y, Z)


def makeform(root, fields, def_vals):
    entries = []
    rows = tk.Frame(root, width=100)
    i=0
    for field in fields:
        lab = tk.Label(rows, width=50, text=field, anchor='w')
        ent = tk.Entry(rows)
        ent.insert(tk.END, def_vals[i])
        lab.pack()
        ent.pack(fill=tk.X)
        entries.append((field, ent))
        i+=1

    rows.pack(side=tk.TOP, padx=5, pady=5)
    return entries