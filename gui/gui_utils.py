import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.figure import Figure


def plot_fig(figure):
    """ Create a new tk window instance with a figure ploted in a canvas
    
    :param figure: figure to plot 
    :return: canvas object
    """
    root = tk.Tk()
    root.title("")
    root.geometry("600x600")
    root.wm_iconbitmap('gui/logo-lm2-f_0.ico')
    canvas = FigureCanvasTkAgg(figure, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
    return canvas

def popupmsg(msg):
    """Instantiate a popup with a message and an "ok" button.
    source: https://pythonprogramming.net/tkinter-popup-message-window/

    :param msg: pop-up string message
    """
    popup = tk.Tk()
    popup.wm_title("Warning")
    label = ttk.Label(popup, text=msg, font=("Helvetica", 10))
    label.pack(side="top", fill="x", pady=10)
    ok_button = ttk.Button(popup, text="Ok", command=popup.destroy)
    ok_button.pack()
    popup.mainloop()


def makeform(root, fields, def_vals, pos=tk.TOP):
    """Create a form with several entries.
    source: https://www.python-course.eu/tkinter_entry_widgets.php

    :param root: tk Frame object to put the form in
    :param fields: field name strings (list)
    :param def_vals: default values string (list)
    :param pos: tkinter constant position value (tk.TOP, LEFT, RIGHT, BOTTOM)
    :return: list with each field name (column 0) and entry object (column 1)
    """
    entries = []
    rows = tk.Frame(root, width=100)
    i = 0
    for field in fields:
        lab = tk.Label(rows, width=50, text=field, anchor='w')
        ent = tk.Entry(rows)
        ent.insert(tk.END, def_vals[i])
        lab.pack()
        ent.pack(fill=tk.X)
        entries.append((field, ent))
        i += 1

    rows.pack(side=pos, padx=5, pady=5)
    return entries
