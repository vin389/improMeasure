import os
import tkinter as tk

def listFilesAndTimes(directory=''):
    # if directory is empty or len is 1, ask user to select from file dialog
    if len(directory) <= 1:
        root = tk.Tk()
        root.withdraw()
        root.lift()
        root.wm_attributes("-topmost", True)
        dir_path = tk.filedialog.askdirectory()
    if len(dir_path) <= 1:
        return [], []
    # create a list of file names (str) 
    #    and a list of file times (int)
    files = []
    times = []
    for filename in os.listdir(dir_path):
        fullpath = os.path.join(dir_path, filename)
        if os.path.isfile(fullpath):
            files.append(filename)
            times.append(os.path.getmtime(fullpath))
#            print(filename, '\t', os.path.getmtime(filename))
    return files, times

# f, t = listsFilesAndTimes()



