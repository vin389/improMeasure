# import frequently used libraries
import numpy as np
import cv2 as cv
import os, sys, glob, time, re
# try to import matplotlib. if fails, display warning message.
try:
    import matplotlib.pyplot as plt
    matplotlib_installed = True
except ImportError:
    print("# Warning: matplotlib is not installed. Some functions may not work.")
    matplotlib_installed = False
# try to import pandas. if fails, display warning message.
try:
    import pandas as pd
    pandas_installed = True
except ImportError:
    print("# Warning: pandas is not installed. Some functions may not work.")
    pandas_installed = False
# try to import scipy. if fails, display warning message.
try:
    import scipy
    scipy_installed = True
except ImportError:
    print("# Warning: scipy is not installed. Some functions may not work.")
    scipy_installed = False




# import numpy 
import cv2
import os, sys, glob, time, re


import tkinter as tk
# Create a tk window 
tk_root = tk.Tk()
# add 3 labels to the window
tk.Label(tk_root, text="Label 1").pack()
tk.Label(tk_root, text="Label 2").pack()
tk.Label(tk_root, text="Label 3").pack()
# add 3 text boxes to the window, also assign them to variables
entry1 = tk.Entry(tk_root)
entry1.pack()
entry2 = tk.Entry(tk_root)
entry2.pack()
entry3 = tk.Entry(tk_root)
entry3.pack()
# make entry3 read-only
entry3.config(state='readonly')
# add a button to the window, which has an event function
def on_button_click():
    entry3.config(state='normal')
    entry3.delete(0, tk.END)
    entry3.insert(0, entry1.get() + entry2.get())
    entry3.config(state='readonly')
button = tk.Button(tk_root, text="Click me", command=on_button_click)
button.pack()
# add a button named "Quit" to the window, which closes the window
quit_button = tk.Button(tk_root, text="Quit", command=tk_root.quit)
quit_button.pack()
# run the window
tk_root.mainloop()
# close the window
try:
    tk_root.destroy()
except:
    print("# End of program.")

