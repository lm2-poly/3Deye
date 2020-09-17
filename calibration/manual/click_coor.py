from tkinter import filedialog
from tkinter import *
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk
import numpy as np


if __name__ == "__main__":
    root = Tk()

    #setting up a tkinter canvas with scrollbars
    frame = Frame(root, bd=2, relief=SUNKEN)
    frame.grid_rowconfigure(0, weight=1)
    frame.grid_columnconfigure(0, weight=1)
    xscroll = Scrollbar(frame, orient=HORIZONTAL)
    xscroll.grid(row=1, column=0, sticky=E+W)
    yscroll = Scrollbar(frame)
    yscroll.grid(row=0, column=1, sticky=N+S)
    canvas = Canvas(frame, bd=0, xscrollcommand=xscroll.set, yscrollcommand=yscroll.set)
    canvas.grid(row=0, column=0, sticky=N+S+E+W)
    xscroll.config(command=canvas.xview)
    yscroll.config(command=canvas.yview)
    frame.pack(fill=BOTH,expand=1)

    #adding the image
    File = askopenfilename(parent=root, initialdir="C:/Users/breum/Desktop/2020-08-26",title='Choose an image.')
    img = ImageTk.PhotoImage(Image.open(File))
    canvas.create_image(0,0,image=img,anchor="nw")
    canvas.config(scrollregion=canvas.bbox(ALL))
    coord_list = []
    #function to be called when mouse is clicked
    def printcoords(event):
        #outputting x and y coords to console
        coord_list.append([event.x,event.y])
        print(coord_list)
        np.savetxt("coord_list", coord_list)
        x1, y1 = (event.x - 5), (event.y - 5)
        x2, y2 = (event.x + 5), (event.y + 5)
        canvas.create_oval(x1, y1, x2, y2, fill='red')
        
    #mouseclick event
    canvas.bind("<Button 1>",printcoords)

    root.mainloop()