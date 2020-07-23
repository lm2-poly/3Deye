import tkinter as tk
from tkinter import ttk
from data_treat.reconstruction_3d import reconstruct_3d
from data_treat.make_report import make_report, load_data
from data_treat.data_pp import get_init_angle, get_impact_position, get_velocity
import os
from gui.gui_utils import makeform, popupmsg


def ana_tab(root,frame, notebook, cam_top, cam_left, traj_3d):
    show_traj = tk.IntVar()
    is_batch = tk.IntVar()

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

    batch_options = tk.Frame(frame)
    batch_label = tk.Label(batch_options, text="Batch folder path:")
    batch_label.pack()
    batch_folder = tk.Entry(batch_options, width=100)
    batch_folder.pack()
    batch_warning = tk.Label(batch_options,
                             text="Each folder to analyse in the batch folder should contain two folders with "
                                  "the name specified for the top and left camera above\nAll the report files will"
                                  "be saved in a 'RESULT' folder created in the batch folder.")
    batch_warning.pack()
    batch_switch = tk.Checkbutton(option_box, text="Batch mode", variable=is_batch,
                                  command=(lambda e=is_batch, bo=batch_options: batch_option_active(e, bo)))


    w = ttk.Combobox(option_box, values=['No perspective', 'Perspective simple', 'Perspective optimized'])
    w.bind("<<ComboboxSelected>>", (lambda val=w.get(), camf=cam_factors: method_change(val, camf)))
    w.insert(tk.END, 'Perspective simple')
    cb = tk.Checkbutton(option_box, text="Show detected points", variable=show_traj)
    thres = makeform(option_box, ['Detection threshold'], ['10.'])

    exp_params_fr = tk.Frame(frame)
    exp_param = makeform(exp_params_fr, ['Shot type', 'Sample name', 'Input pressure (psi)'],
                   ["0.5mm", "Aluminum 6050", "50"])


    top = makeform(top_cam, ['Calibration folder',"Picture folder", 'First picture ID', 'framerate',
                             'Screen width', 'Screen height', "Acquisition width", "Acquisition height"],
                    ["calibration/res", "tests/single/camTop", "0", '15000',
                     "1280", "800", "1280", "800"])


    titleLeft = tk.Label(left_cam, text="Left camera parameters")
    titleLeft.pack(side=tk.TOP)
    left = makeform(left_cam, ['Calibration folder',"Picture folder", 'First picture ID', 'framerate',
                             'Screen width', 'Screen height', "Acquisition width", "Acquisition height"],
                    ["calibration/res", "tests/single/camLeft", "0", '15000',
                     "1280", "800", "1280", "800"])

    b1 = tk.Button(frame, text='Launch Analysis !',
                   command=(lambda t=top, l=left, n=notebook, wval=w, s=show_traj, ct=cam_top,
                                   cl=cam_left, traj=traj_3d, ratTop=cam_top_factor, ratLeft=cam_left_factor,
                                   bo=is_batch, bf=batch_folder, ep=exp_param, th=thres:
                            launch_analysis(t, l, n, wval,ct, cl, traj, s, ratTop, ratLeft, bo, bf, ep, th)))

    exp_params_fr.pack(side=tk.TOP)
    b1.pack(side=tk.BOTTOM, padx=5, pady=5)
    w.pack(side=tk.LEFT)
    cb.pack(side=tk.RIGHT)
    batch_switch.pack(side=tk.RIGHT)
    option_box.pack(side=tk.BOTTOM)
    warning_label = tk.Label(frame, text="Warning, picture name must be in the following format: 'Name_number.jpg'")
    warning_label.pack(side=tk.BOTTOM)

    top_cam.pack(side=tk.LEFT, padx=5, pady=5)
    left_cam.pack(side=tk.LEFT, padx=5, pady=5)
    cam_frames.pack(side=tk.TOP)


def batch_option_active(switch_val, batch_options):
    if switch_val.get():
        batch_options.pack(side=tk.TOP)
    else:
        batch_options.pack_forget()


def method_change(val, cam_factors):
    if val.widget.get() == 'No perspective':
        cam_factors.pack(side=tk.BOTTOM)

    else:
        cam_factors.pack_forget()


def create_camera(entries, name, cam, pic_to_cm=None):
    cam.load_calibration_file(entries[0][1].get()+'/cam_'+name)
    cam.dir = entries[1][1].get()
    cam.firstPic = int(entries[2][1].get())
    cam.pic_to_cm = 1 / 141.1
    cam.framerate = float(entries[3][1].get())
    cam.camRes = (int(entries[4][1].get()), int(entries[5][1].get()))
    cam.res = (int(entries[6][1].get()), int(entries[7][1].get()))

    if not pic_to_cm is None:
        cam.pic_to_cm = pic_to_cm
    return cam


def launch_analysis(top_entry, left_entry, notebook, method, cam_top, cam_left, traj_3d, show_traj,
                    ratTop, ratLeft, isbatch, batch_folder, exp_param, thres):


    traj_3d.set_exp_params(exp_param[0][1].get(), exp_param[1][1].get(), exp_param[2][1].get())

    if method.get() == "No perspective":
        meth = 'no-persp'
    elif method.get() == "Perspective simple":
        meth = 'persp'
    else:
        meth = 'persp-opti'

    if isbatch.get():
        foldList = os.listdir(batch_folder.get())
        ana_fold = batch_folder.get()+'/'
        if 'RESULTS' in foldList:
            foldList.remove('RESULTS')
        os.system('cd ' + ana_fold + ' && mkdir RESULTS')
    else:
        foldList  = ['']
        ana_fold = ''

    notebook.tab(3, state='normal')

    for elem in foldList:
        print("************ "+elem)
        create_camera(top_entry, 'top', cam_top, float(ratTop.get()))
        create_camera(left_entry, 'left', cam_left, float(ratLeft.get()))
        if not elem == '':
            cam_top.dir = ana_fold + elem + '/' + cam_top.dir
            cam_left.dir = ana_fold + elem + '/' + cam_left.dir
        cam_top.set_crop_size()
        cam_left.set_crop_size()

        X, Y, Z, timespan = reconstruct_3d(cam_top, cam_left,
                                           splitSymb="_", numsplit=-1, method=meth,
                                           plotTraj=show_traj.get(), plot=not(isbatch.get()), threshold=float(thres[0][1].get()))
        traj_3d.set_trajectory(timespan, X, Y, Z)

        if isbatch.get():
            alpha = get_init_angle(X, Y, Z, timespan, cam_top, cam_left, plot=False, saveDir=ana_fold+'RESULTS\\'+elem+'-')
            xi, yi, zi = get_impact_position(X, Y, Z, cam_left, cam_top, plot=False, saveDir=ana_fold+'RESULTS\\'+elem+'-')
            Vinit, Vend = get_velocity(timespan, X, Y, Z, thres=1.1, plot=False, saveDir=ana_fold+'RESULTS\\'+elem+'-')
            make_report(traj_3d, alpha, Vinit, Vend, [xi, yi, zi], cam_top, cam_left,
                        ana_fold+'RESULTS/'+elem+'.txt', "data_treat/report_template.txt")
