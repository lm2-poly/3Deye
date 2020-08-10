import tkinter as tk
from data_treat.make_report import load_data
from gui.gui_utils import popupmsg


def load_tab(root, frame, cam_top, cam_left, traj_3d, notebook):
    """Setup the loading tab

    :param root: tk root window
    :param frame: calibration tab frame object
    :param cam_top,cam_left: top and left camera objects
    :param traj_3d: Experiment object
    :param notebook: notebook object the tab belongs to
    """
    lab = tk.Label(frame, width=50, text='Previous analysis report file path:', anchor='w')
    ent = tk.Entry(frame, width=50)
    ent.insert(tk.END, 'Trajectory.txt')
    b1 = tk.Button(frame, text='Load data',
                   command=(
                       lambda n=notebook, f=ent, tra=traj_3d, ct=cam_top, cl=cam_left:
                       start_load(n, f, tra, ct, cl)))

    lab.pack()
    ent.pack(fill=tk.X)
    b1.pack()


def start_load(notebook, f, tra, ct, cl):
    """loads an analysis by initializing the cameras and experiment objects

    :param notebook: notebook object the tab belongs to (enables the pp mode)
    :param f: form entries
    :param tra: Experiment object
    :param ct,cl: camera_top and left objects
    """

    try:
        load_data(f.get(), tra, ct, cl)
    except:
        notebook.tab(3, state='disabled')
        popupmsg('File not found...')
    else:
        notebook.tab(3, state='normal')
        popupmsg('Data successfully loaded !')