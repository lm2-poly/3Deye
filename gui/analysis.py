import tkinter as tk
from tkinter import ttk
from data_treat.reconstruction_3d import reconstruct_3d
from data_treat.make_report import make_report
from data_treat.data_pp import get_init_angle, get_impact_position, get_velocity
import os
from gui.gui_utils import makeform
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.figure import Figure
from PIL import Image
import glob
import numpy as np


def ana_tab(root,frame, notebook, cam_top, cam_left, traj_3d):
    """Setup the analysis tab

    :param root: root tk window
    :param frame: frame of the tab to draw in
    :param notebook: notebook object the tab belongs to
    :param cam_top,cam_left: top and left camera objects
    :param traj_3d: Experiment object
    """
    show_traj = tk.IntVar()
    is_batch = tk.IntVar()

    cam_frames = tk.Frame(frame)
    top_cam = tk.Frame(cam_frames, width=250)
    left_cam = tk.Frame(cam_frames, width=250)

    option_box = tk.Frame(frame)

    cam_factors = tk.Frame(frame)
    cam_factors.hidden = 1
    cam_top_lab = tk.Label(cam_factors, text="Top pixel to cm ratio")
    cam_top_lab.pack(side=tk.LEFT)
    cam_top_factor = tk.Entry(cam_factors)
    cam_top_factor.insert(tk.END, '{:.04e}'.format(1 / 141.1))
    cam_top_factor.pack(side=tk.LEFT)

    cam_left_lab = tk.Label(cam_factors, text="Left pixel to cm ratio")
    cam_left_lab.pack(side=tk.LEFT)
    cam_left_factor = tk.Entry(cam_factors)
    cam_left_factor.insert(tk.END, '{:.04e}'.format(1 / 148.97))
    cam_left_factor.pack(side=tk.LEFT)

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
    batch_params = tk.Button(batch_options, text="Batch set-up", command=(lambda t3d=traj_3d: set_pp_params(t3d)))
    batch_params.pack(side=tk.RIGHT)
    batch_switch = tk.Checkbutton(option_box, text="Batch mode", variable=is_batch,
                                  command=(lambda e=is_batch, bo=batch_options: batch_option_active(e, bo)))


    w = ttk.Combobox(option_box, values=['No perspective', 'Perspective simple', 'Perspective optimized'])
    w.bind("<<ComboboxSelected>>", (lambda val=w.get(), camf=cam_factors: method_change(val, camf)))
    w.insert(tk.END, 'Perspective simple')
    cb = tk.Checkbutton(option_box, text="Show detected points", variable=show_traj)

    exp_params_fr = tk.Frame(frame)
    exp_param = makeform(exp_params_fr, ['Shot type', 'Sample name', 'Input pressure (psi)'],
                   ["0.5mm", "Aluminum 6050", "50"])

    top_base = tk.Frame(top_cam)
    titleTop = tk.Label(top_base, text="Top camera parameters")
    titleTop.pack(side=tk.LEFT)
    top = makeform(top_cam, ['Calibration folder',"Picture folder", 'First picture ID', 'framerate',
                             'Screen width', 'Screen height', "Acquisition width", "Acquisition height",
                             'Detection threshold'],
                    ["calibration/res", "tests/single/camTop", "0", '15000',
                     "1280", "800", "1280", "800", "20."], pos=tk.BOTTOM)
    b1 = tk.Button(top_base, text='Set mask', command=(lambda ct=cam_top, t=top: set_mask(ct, t)))
    b1.pack(side=tk.RIGHT)
    top_base.pack(side=tk.TOP)

    left_base = tk.Frame(left_cam)
    titleLeft = tk.Label(left_base, text="Left camera parameters")
    titleLeft.pack(side=tk.LEFT)
    left = makeform(left_cam, ['Calibration folder',"Picture folder", 'First picture ID', 'framerate',
                             'Screen width', 'Screen height', "Acquisition width", "Acquisition height",
                               'Detection threshold'],
                    ["calibration/res", "tests/single/camLeft", "0", '15000',
                     "1280", "800", "1280", "800", '20.'], pos=tk.BOTTOM)
    b1 = tk.Button(left_base, text='Set mask', command=(lambda cl=cam_left, l=left: set_mask(cl, l)))
    b1.pack(side=tk.RIGHT)
    left_base.pack(side=tk.TOP)

    b1 = tk.Button(frame, text='Launch Analysis !',
                   command=(lambda t=top, l=left, n=notebook, wval=w, s=show_traj, ct=cam_top,
                                   cl=cam_left, traj=traj_3d, ratTop=cam_top_factor, ratLeft=cam_left_factor,
                                   bo=is_batch, bf=batch_folder, ep=exp_param:
                            launch_analysis(t, l, n, wval,ct, cl, traj, s, ratTop, ratLeft, bo, bf, ep)))

    exp_params_fr.pack(side=tk.TOP)
    b1.pack(side=tk.BOTTOM, padx=5, pady=5)
    w.pack(side=tk.LEFT)
    cb.pack(side=tk.RIGHT)
    batch_switch.pack(side=tk.RIGHT)
    option_box.pack(side=tk.BOTTOM)
    warning_label = tk.Label(frame, text="Warning, picture name must be in the following format: 'Name_number.jpg'")
    warning_label.pack(side=tk.BOTTOM)

    top_cam.pack(side=tk.LEFT, padx=5, pady=5)
    left_cam.pack(side=tk.RIGHT, padx=5, pady=5)
    cam_frames.pack(side=tk.TOP)


def set_mask(cam, form):
    """Generate the subwindow to set a camera mask

    :param cam: camera object
    :param form: camera form values to get the first picture path
    """
    mask_w = tk.IntVar()
    mask_h = tk.IntVar()
    root = tk.Tk()
    root.title("Set mask")
    root.geometry("600x600")
    root.wm_iconbitmap('gui/logo-lm2-f_0.ico')
    cam_pics = glob.glob(form[1][1].get() + "/*.tif")
    if len(cam_pics) == 0:
        cam_pics = glob.glob(form[1][1].get() + "/*.jpg")
    im_act = Image.open(cam_pics[0])
    fig = Figure(figsize=(5,4), dpi=100)
    im_act = np.array(im_act)
    fig_plot = fig.add_subplot(111).imshow(im_act, cmap='gray')
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
    toolbar = NavigationToolbar2Tk(canvas, root)
    toolbar.update()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    wd_lab = tk.Label(root, text="width")
    mask_val_w = tk.Scale(root, from_=0, to=im_act.shape[1], orient=tk.HORIZONTAL, variable=mask_w,
                        command=(lambda ma=mask_w, p=im_act, c=canvas: update_fig(ma, p, c, 0)))
    wd_lab.pack(side=tk.LEFT)
    mask_val_w.pack(side=tk.LEFT)

    hi_lab = tk.Label(root, text="height")
    mask_val_h = tk.Scale(root, from_=0, to=im_act.shape[0], orient=tk.HORIZONTAL, variable=mask_h,
                        command=(lambda ma=mask_h, p=im_act, c=canvas: update_fig(ma, p, c, 1)))
    hi_lab.pack(side=tk.LEFT)
    mask_val_h.pack(side=tk.LEFT)

    b1 = tk.Button(root, text='Set mask',
                   command=(lambda r=root, c=cam, mw=mask_val_w, mh=mask_val_h: set_cam_mask(r, c, mw, mh)))
    b1.pack()


def set_cam_mask(root, cam, mask_w, mask_h):
    """sets the camera mask and closes the mask selection window"""
    cam.set_mask(int(mask_w.get()), int(mask_h.get()))
    root.destroy()


def update_fig(mask, im_act, canvas, ind):
    """update the mask selection picture as the slider moves

    :param mask: mask value
    :param im_act: plotted image ndarray
    :param canvas: canvas to plot the modified image onto
    :param ind: 0 for the mask to act on the pictur width, 1 for the picture height
    """
    pic_act = np.copy(im_act)
    if ind == 0:
        pic_act[:, :int(mask)] = 0
    elif ind == 1:
        pic_act[im_act.shape[0] - int(mask):, :] = 0
    canvas.figure.clear()
    canvas.figure.add_subplot(111).imshow(pic_act, cmap='gray')
    canvas.draw()


def set_pp_params(traj_3d):
    """Generate the batch PP parameter subwindow

    :param traj_3d: Experience object to change
    """
    param_win = tk.Tk()
    param_win.title("Batch post-processing parameters")
    param_win.geometry("600x400")
    param_win.wm_iconbitmap('gui/logo-lm2-f_0.ico')

    vel_frame = tk.Frame(param_win)
    title = tk.Label(vel_frame, text="Velocity determination parameters")
    title.pack(side=tk.TOP)
    vels = makeform(vel_frame, ['Velocity detection factor', 'Initial index', 'Minimum number of points'],
                    [traj_3d.vel_det_fac, traj_3d.vel_init_ind, traj_3d.vel_min_pt])

    angle_frame = tk.Frame(param_win)
    title = tk.Label(angle_frame, text="Angle determination parameters")
    title.pack(side=tk.TOP)
    angles = makeform(angle_frame, ['Initial index', 'End index'], [traj_3d.ang_min_ind, traj_3d.ang_end_ind])

    pos_frame = tk.Frame(param_win)
    title = tk.Label(angle_frame, text="Impact position parameters")
    title.pack(side=tk.TOP)
    pos = makeform(angle_frame, ['Threshold'], [traj_3d.imp_thres])

    vel_frame.pack(side=tk.TOP)
    angle_frame.pack(side=tk.TOP)
    pos_frame.pack(side=tk.TOP)

    b1 = tk.Button(param_win, text='Ok', command=(lambda t3d=traj_3d, v=vels, a=angles, p=pos, f=param_win:
                                                             save_pp_params(t3d, v, a, p, f)))
    b1.pack(side=tk.BOTTOM, padx=5, pady=5)


def save_pp_params(traj_3d, vels, angles, pos, param_win):
    """saves PP parameters in an Experiment object and closes the window

    :param traj_3d: Experiment object
    :param vels: velocity determination parameters
    :param angles: angle determination parameters
    :param pos: impact position determination parameters
    :param param_win: PP parameter window to shut down
    :return:
    """
    traj_3d.vel_det_fac = float(vels[0][1].get())
    traj_3d.vel_init_ind = int(vels[1][1].get())
    traj_3d.vel_min_pt = int(vels[2][1].get())
    traj_3d.ang_min_ind = int(angles[0][1].get())
    traj_3d.ang_end_ind = int(angles[1][1].get())
    traj_3d.imp_thres = float(pos[0][1].get())
    param_win.destroy()


def batch_option_active(switch_val, batch_options):
    """Check if the batch mode was activate and plots the batch options accordingly

    :param switch_val: batch mode switch
    :param batch_options: batch option tk Frame
    """
    if switch_val.get():
        batch_options.pack(side=tk.TOP)
    else:
        batch_options.pack_forget()


def method_change(val, cam_factors):
    """Checks if the method was changed to no-perspective and changes the GUI accordingly

    :param val: method choice tk combobox object
    :param cam_factors: method parameters to display
    """
    if val.widget.get() == 'No perspective':
        cam_factors.pack(side=tk.BOTTOM)

    else:
        cam_factors.pack_forget()


def create_camera(entries, name, cam, pic_to_cm=None):
    """Create a camera object based on the values filled in the form

    :param entries: form entries
    :param name: camera name (top or left)
    :param cam: camera object
    :param pic_to_cm: picture to cm ratio if the no-perspective mode is used
    :return: initialized camera object
    """
    cam.load_calibration_file(entries[0][1].get()+'/cam_'+name)
    cam.dir = entries[1][1].get()
    cam.firstPic = int(entries[2][1].get())
    cam.pic_to_cm = 1 / 141.1
    cam.framerate = float(entries[3][1].get())
    cam.camRes = (int(entries[4][1].get()), int(entries[5][1].get()))
    cam.res = (int(entries[6][1].get()), int(entries[7][1].get()))
    cam.cam_thres = float(entries[8][1].get())
    if not pic_to_cm is None:
        cam.pic_to_cm = pic_to_cm
    return cam


def launch_analysis(top_entry, left_entry, notebook, method, cam_top, cam_left, traj_3d, show_traj,
                    ratTop, ratLeft, isbatch, batch_folder, exp_param):
    """Launch a 3D trajectory analysis

    :param top_entry,left_entry: form entries for the top and left cameras
    :param notebook: GUI notebook object (to enable the pp tab at the end
    :param method: analysis method combobox object
    :param cam_top,cam_left: camera top and left objects
    :param traj_3d: Experiment object
    :param show_traj: Trajectory display checkbox object
    :param ratTop,ratLeft: pixel to cm ratio form objects for the top and left cameras
    :param isbatch: checkbox object enabling (or not) batch mode
    :param batch_folder: batch folder path form object
    :param exp_param: experimental parameter form object
    :return:
    """

    traj_3d.set_exp_params(exp_param[0][1].get(), exp_param[1][1].get(), exp_param[2][1].get(), "Trajectory.txt")

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
                                           plotTraj=show_traj.get(), plot=not(isbatch.get()))
        traj_3d.set_trajectory(timespan, X, Y, Z)

        if isbatch.get():
            alpha = get_init_angle(X, Y, Z, timespan, cam_top, cam_left, plot=False,
                                   saveDir=ana_fold+'RESULTS\\'+elem+'-', init=traj_3d.ang_min_ind, end=traj_3d.ang_end_ind)
            xi, yi, zi = get_impact_position(X, Y, Z, cam_left, cam_top, plot=False,
                                             saveDir=ana_fold+'RESULTS\\'+elem+'-', threshold= traj_3d.imp_thres)
            Vinit, Vend = get_velocity(timespan, X, Y, Z, thres=traj_3d.vel_det_fac, plot=False,
                                       saveDir=ana_fold+'RESULTS\\'+elem+'-', init=traj_3d.vel_init_ind, pt_num=traj_3d.vel_min_pt)
            traj_3d.set_pp(alpha, Vinit, Vend, [xi, yi, zi])
            traj_3d.save_dir = ana_fold+'RESULTS/'+elem+'.txt'
            make_report(traj_3d, cam_top, cam_left, "data_treat/report_template.txt")