import tkinter as tk
from tkinter import ttk


def popupmsg(msg):
    popup = tk.Tk()
    popup.wm_title("Warning")
    label = ttk.Label(popup, text=msg, font=("Helvetica", 10))
    label.pack(side="top", fill="x", pady=10)
    B1 = ttk.Button(popup, text="Ok", command=popup.destroy)
    B1.pack()
    popup.mainloop()


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