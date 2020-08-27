import tkinter as tk
from calibration.main import calibrate_stereo
from gui.gui_utils import makeform, popupmsg


def calib_tab(root,frame):
    """Setup teh calibration tab content

    :param root: tk root window
    :param frame: calibration tab frame object
    :return:
    """
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


    # ents = makeform(frame, ['Top camera chessboard picture folder',
    #                         'Left camera chessboard picture folder',
    #                         'Top camera sample position picture',
    #                         'Left camera sample position picture',
    #                         'Output calibration files folder path'],
    #                 ["calibration/lens_dist_calib_top",
    #               "calibration/lens_dist_calib_left",
    #               "calibration/sources/calib_checker_top.jpg",
    #               "calibration/sources/calib_checker_left.jpg",
    #                  "calibration/res"])
    ents = makeform(frame, ['Top camera chessboard picture folder',
                            'Left camera chessboard picture folder',
                            'Top camera sample position picture',
                            'Left camera sample position picture',
                            'Output calibration files folder path'],
                    ["C:/Users/breum/Desktop/2020-08-26/calib/LEFT",
                     "C:/Users/breum/Desktop/2020-08-26/calib/TOP",
                     "C:/Users/breum/Desktop/2020-08-26/calib_30deg_top.tif",
                     "C:/Users/breum/Desktop/2020-08-26/calib_30deg_left.tif",
                     "C:/Users/breum/Desktop/2020-08-26/calib_res/30deg"])

    b1 = tk.Button(frame, text='Calibrate !',
                   command=(lambda e=ents: launch_calib(e, numcase, lencase)))
    chess_param.pack()
    b1.pack()


def launch_calib(entries, numcase, lencase):
    """Launch calibration procedure

    :param entries: calibration form entries
    :param numcase: number of checkboard case form object
    :param lencase: length of a checkboard case form object
    """
    try:
        calibrate_stereo(entries[0][1].get(),
                         entries[1][1].get(),
                         entries[2][1].get(),
                         entries[3][1].get(),
                         entries[4][1].get(),
                         chess_dim=int(numcase.get()), chess_case_len=float(lencase.get()))
    except:
        popupmsg("Calibration failed !")
