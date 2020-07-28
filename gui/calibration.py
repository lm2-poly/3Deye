import tkinter as tk
from calibration.main import calibrate_stereo
from gui.gui_utils import makeform, popupmsg


def calib_tab(root,frame):
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

    b1 = tk.Button(frame, text='Calibrate !',
                   command=(lambda e=ents: launch_calib(e, frame, numcase, lencase)))
    chess_param.pack()
    b1.pack()


def launch_calib(entries, frame, numcase, lencase):

    calibrate_stereo(entries[0][1].get(),
                     entries[1][1].get(),
                     entries[2][1].get(),
                     entries[3][1].get(),
                     entries[4][1].get(),
                     chess_dim=int(numcase.get()), chess_case_len=float(lencase.get()))


def all_children(window) :
    _list = window.winfo_children()

    for item in _list :
        if item.winfo_children() :
            _list.extend(item.winfo_children())

    return _list
