import numpy as np
import tkinter as tk
import tkinter.filedialog as fd
from ImproToolTip import ImproToolTip

# The class name is Mss0. 
# This is a tkinter GUI part of code. This is a program with Tkinter GUI.
# In GUI window, use grid for layout.
# In grid row 0, at left side, there is a button titled 'Working directory ...', 
# and is named bt_workDir, 
# right side is a text entry, named tx_workDir, that allows user to key-in. 
# The width of tx_workDir is not fixed. It is from the right side of the button to the right side of the window.
# If user clicks button bt_workDir, a tk file dialog pops up for user to select a directory.
# If user does not cancel the file dialog, the selected directory path is displayed at tx_workDir. 
# Use ImproToolTip to add a tooltip to tx_workDir.
# If mouse hovers over tx_workDir, a tooltip is temporarily displayed, with text of "Select working directory or enter it here"

class Mss0:
    def __init__(self, master=None):
        self.master = master
        self.master.title("MSS0 Example")
        self.master.geometry("400x200")

        # Create a button to select working directory
        self.bt_workDir = tk.Button(self.master, text="Working directory ...", command=self.select_working_directory)
        self.bt_workDir.grid(row=0, column=0, padx=5, pady=5, sticky='w')

        # Create an entry to display the working directory
        # The width of the entry is not fixed. It is from the right side of the button to the right side of the window
        self.master.grid_columnconfigure(1, weight=1)
        self.tx_workDir = tk.Entry(self.master)
        self.tx_workDir.grid(row=0, column=1, padx=5, pady=5, sticky='ew')

        # Add tooltip to the entry
        ImproToolTip(self.tx_workDir, "Select working directory or enter it here")

    def select_working_directory(self):
        directory = fd.askdirectory(title="Select Working Directory")
        if directory:
            self.tx_workDir.delete(0, tk.END)  # Clear the entry
            self.tx_workDir.insert(0, directory)  # Insert the selected directory path


if __name__ == "__main__":
    root = tk.Tk()
    app = Mss0(master=root)
    root.mainloop()

