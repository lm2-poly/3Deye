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
    root.geometry("1000x650")
    style = ttk.Style(root)
    style.configure("lefttab.TNotebook", tabposition="wn")

    notebook = ttk.Notebook(root, style="lefttab.TNotebook")

    calib_frame = tk.Frame(notebook, width=800, height=600)
    load_frame = tk.Frame(notebook, width=800, height=600)
    ana_frame = tk.Frame(notebook, width=800, height=600)
    pp_frame = tk.Frame(notebook, width=800, height=600)
    root.update_idletasks()

    calib_tab(root, calib_frame)
    load_tab(root, load_frame, cam_top, cam_left, traj_3d, notebook)
    ana_tab(root, ana_frame, notebook, cam_top, cam_left, traj_3d)
    pp_tab(root, pp_frame, cam_top, cam_left, traj_3d)

    notebook.add(calib_frame, text="Calibration")
    notebook.add(load_frame, text="Load analysis")
    notebook.add(ana_frame, text="Analysis")
    notebook.add(pp_frame, text="Post processing", state="disabled")

    notebook.grid(row=0, column=0, sticky="n")

    root.mainloop()


def calib_tab(root,frame):
    pic = tk.Frame(frame, width=500, height=350,
                   highlightbackground="black", highlightthickness=1.)

    chess_param = tk.Frame(frame)
    numcase_lab = tk.Label(chess_param, text="Number of chessboard cases -1")
    numcase = tk.Entry(chess_param)
    numcase.insert(tk.END, 7)
    lencase_lab = tk.Label(chess_param, text="Cases real life length")
    lencase = tk.Entry(chess_param)
    lencase.insert(tk.END, 0.4375)

    numcase_lab.pack()
    numcase.pack()
    lencase_lab.pack()
    lencase.pack()
    chess_param.pack(side=tk.RIGHT)

    ents = makeform(frame, ['Top camera chessboard picture folder',
                            'Left camera chessboard picture folder',
                            'Top camera sample position picture',
                            'Left camera sample position picture',
                            'Output calibration files folder path'],
                    ["calibration/lens_dist_calib_top",
                  "calibration/lens_dist_calib_left",
                  "calibration/sources/calib_checker_top.jpg",
                  "calibration/sources/calib_checker_left.jpg",
                     "calibration/res"])

    #frame.bind('<Return>', (lambda event, e=ents: launch_calib(e, pic)))
    b1 = tk.Button(frame, text='Calibrate !',
                   command=(lambda e=ents: launch_calib(e, pic, frame, numcase, lencase)))
    pic.pack(side=tk.BOTTOM)
    b1.pack(side=tk.BOTTOM, padx=5, pady=5)


def load_tab(root, frame, cam_top, cam_left, traj_3d, notebook):
    lab = tk.Label(frame, width=50, text='Previous analysis report file path:', anchor='w')
    ent = tk.Entry(frame, width=50)
    ent.insert(tk.END, 'Trajectory.txt')
    b1 = tk.Button(frame, text='Load data',
                   command=(
                       lambda n=notebook, f=ent.get(), tra=traj_3d, ct=cam_top, cl=cam_left:
                       start_load(n, f, tra, ct, cl)))

    lab.pack()
    ent.pack(fill=tk.X)
    b1.pack()


def start_load(notebook, f, tra, ct, cl):
    notebook.tab(3, state='normal')
    load_data(f, tra, ct, cl)
    popupmsg('Data successfully loaded !')


def popupmsg(msg):
    popup = tk.Tk()
    popup.wm_title("Warning")
    label = ttk.Label(popup, text=msg, font=("Helvetica", 10))
    label.pack(side="top", fill="x", pady=10)
    B1 = ttk.Button(popup, text="Ok", command=popup.destroy)
    B1.pack()
    popup.mainloop()


def ana_tab(root,frame, notebook, cam_top, cam_left, traj_3d):
    show_traj = tk.IntVar()
    cam_frames = tk.Frame(frame)
    top_cam = tk.Frame(cam_frames, width=250)
    left_cam = tk.Frame(cam_frames, width=250)

    titleTop = tk.Label(top_cam, text="Top camera parameters")
    titleTop.pack(side=tk.TOP)
    option_box = tk.Frame(frame)

    cam_factors = tk.Frame(frame)
    cam_factors.hidden = 1
    cam_top_lab = tk.Label(cam_factors, text="Top pixel to cm ratio")
    cam_top_lab.pack()
    cam_top_factor = tk.Entry(cam_factors)
    cam_top_factor.insert(tk.END, '{:.04e}'.format(1 / 141.1))
    cam_top_factor.pack()

    cam_left_lab = tk.Label(cam_factors, text="Left pixel to cm ratio")
    cam_left_lab.pack()
    cam_left_factor = tk.Entry(cam_factors)
    cam_left_factor.insert(tk.END, '{:.04e}'.format(1 / 148.97))
    cam_left_factor.pack()

    w = ttk.Combobox(option_box, values=['No perspective', 'Perspective simple', 'Perspective optimized'])
    w.bind("<<ComboboxSelected>>", (lambda val=w.get(), camf=cam_factors: method_change(val, camf)))
    w.insert(tk.END, 'Perspective simple')
    cb = tk.Checkbutton(option_box, text="Show detected points", variable=show_traj)

    top = makeform(top_cam, ['Calibration folder',"Picture folder", 'First picture ID', 'framerate',
                             'Screen width', 'Screen height', "Acquisition width", "Acquisition height"],
                    ["calibration/res", "camTop", "0", '15000',
                     "500", "500", "500", "500"])


    titleLeft = tk.Label(left_cam, text="Left camera parameters")
    titleLeft.pack(side=tk.TOP)
    left = makeform(left_cam, ['Calibration folder',"Picture folder", 'First picture ID', 'framerate',
                             'Screen width', 'Screen height', "Acquisition width", "Acquisition height"],
                    ["calibration/res", "camLeft", "0", '15000',
                     "500", "500", "500", "500"])

    b1 = tk.Button(frame, text='Launch Analysis !',
                   command=(lambda t=top, l=left, n=notebook, wval=w, s=show_traj, ct=cam_top,
                                   cl=cam_left, traj=traj_3d, ratTop=cam_top_factor, ratLeft=cam_left_factor:
                            launch_analysis(t, l, n, wval,ct, cl, traj, s, ratTop, ratLeft)))

    b1.pack(side=tk.BOTTOM, padx=5, pady=5)
    w.pack(side=tk.LEFT)
    cb.pack(side=tk.RIGHT)
    option_box.pack(side=tk.BOTTOM)
    warning_label = tk.Label(frame, text="Warning, picture name must be in the following format: 'Name_number.jpg'")
    warning_label.pack(side=tk.BOTTOM)

    top_cam.pack(side=tk.LEFT, padx=5, pady=5)
    left_cam.pack(side=tk.LEFT, padx=5, pady=5)
    cam_frames.pack(side=tk.TOP)

def method_change(val, cam_factors):
    if val.widget.get() == 'No perspective':
        cam_factors.pack(side=tk.BOTTOM)

    else:
        cam_factors.pack_forget()


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
    make_report(traj_3d.t, traj_3d.X, traj_3d.Y, traj_3d.Z, alpha, Vinit, Vend, [xi, yi, zi], cam_top, cam_left, entries[1][1].get(),
                "data_treat/report_template.txt")
    log += 'Report exported as Report.txt\nTrajectory exported as '+entries[1][1].get()
    T.delete('1.0', tk.END)
    T.insert(tk.END, log)


def launch_calib(entries, pic, frame, numcase, lencase):
    toDel = all_children(pic)
    for item in toDel:
        item.destroy()
    calibrate_stereo(entries[0][1].get(),
                     entries[1][1].get(),
                     entries[2][1].get(),
                     entries[3][1].get(),
                     entries[4][1].get(),
                     chess_dim=int(numcase.get()), chess_case_len=float(lencase.get()), pic=pic)


def all_children(window) :
    _list = window.winfo_children()

    for item in _list :
        if item.winfo_children() :
            _list.extend(item.winfo_children())

    return _list


def create_camera(entries, name, cam, pic_to_cm=None):
    # cam.set_mtx(entries[0][1].get() + "/mtx_"+name)
    # cam.set_dist(entries[0][1].get() + "/dist_"+name)
    # cam.set_R(entries[0][1].get() + "/R_"+name)
    # cam.set_T(entries[0][1].get() + "/T_"+name)
    cam.load_calibration_file(entries[0][1].get()+'/cam_'+name)
    cam.dir = entries[1][1].get()
    cam.firstPic = int(entries[2][1].get())
    cam.pic_to_cm = 1 / 141.1
    cam.framerate = float(entries[3][1].get())
    cam.camRes = (int(entries[4][1].get()), int(entries[5][1].get()))
    cam.res = (int(entries[6][1].get()), int(entries[7][1].get()))
    cam.set_crop_size()
    if not pic_to_cm is None:
        cam.pic_to_cm = pic_to_cm
    return cam


def launch_analysis(top_entry, left_entry, notebook, method, cam_top, cam_left, traj_3d, show_traj, ratTop, ratLeft):
    plt.close()
    create_camera(top_entry, 'top', cam_top, float(ratTop.get()))
    create_camera(left_entry, 'left', cam_left, float(ratLeft.get()))
    notebook.tab(3, state='normal')
    if method.get() == "No perspective":
        meth = 'no-persp'
    elif method.get() == "Perspective simple":
        meth = 'persp'
    else:
        meth = 'persp-opti'

    X, Y, Z, timespan = reconstruct_3d(cam_top, cam_left,
                                       splitSymb="_", numsplit=-1, method=meth, plotTraj=show_traj.get())
    traj_3d.set_trajectory(timespan, X, Y, Z)
    popupmsg("Analysis done")



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