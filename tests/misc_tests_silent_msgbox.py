# the result is incorrect.
# I still don't know how to popup a tkinter messagebox without a ding sound. 

import tkinter as tk
from tkinter import messagebox

root = tk.Tk()
root.withdraw()  # Hide the main window

def show_message_box():
    messagebox.showinfo("Title", "Message text", icon="")

# The message box will appear directly, no need for a button in this case
show_message_box() # Call the function to show the message box immediately

root.mainloop() # Keep the mainloop running for the message box to appear