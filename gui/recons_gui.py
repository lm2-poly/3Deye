import tkinter as tk
from tkinter import ttk
from data_treat.cam import Cam
from data_treat.trajectory import Experiment
from gui.calibration import calib_tab
from gui.load import load_tab
from gui.post_processing import pp_tab
from gui.analysis import ana_tab


def start_gui():
    cam_top = Cam()
    cam_left = Cam()
    traj_3d = Experiment()

    root = tk.Tk()
    root.title("Eye3D")
    root.geometry("1000x750")
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
