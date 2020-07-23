import tkinter as tk
from data_treat.make_report import make_report, load_data
from gui.gui_utils import makeform, popupmsg


def load_tab(root, frame, cam_top, cam_left, traj_3d, notebook):
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
    notebook.tab(3, state='normal')
    load_data(f.get(), tra, ct, cl)
    popupmsg('Data successfully loaded !')