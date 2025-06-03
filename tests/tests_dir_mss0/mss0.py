import tkinter as tk
from tkinter import filedialog
from ImproToolTip import ImproToolTip
import os

class Mss0:
    def __init__(self, root):
        self.root = root
        self.root.title("Mss0 GUI")
        # set window size to 
        self.root.geometry("1400x700")
        self.resizable = (True, True)  # Allow resizing

        # bt_work_dir: Button: 'Working directory ...'
        self.bt_work_dir = tk.Button(root, text="Working directory ...", name="bt_work_dir")
        self.bt_work_dir.grid(row=0, column=0, padx=5, pady=5, sticky='w')
        ImproToolTip(self.bt_work_dir, "Select working directory")
        def select_directory(*args):
            path = filedialog.askdirectory()
            if path:
                self.tx_work_dir.delete(0, tk.END)
                self.tx_work_dir.insert(0, path)
            # trigger check_directory to update text color
            check_directory()
        self.bt_work_dir.bind("<Button-1>", select_directory)

        # tx_work_dir: Entry: Directory path
        self.tx_work_dir = tk.Entry(root, name="tx_work_dir")
        self.tx_work_dir.grid(row=0, column=1, padx=5, pady=5, sticky='we')  # expandable width
        ImproToolTip(self.tx_work_dir, "Enter working diretory here")
        # If user edits the text in the entry, it checks if the directory exists
        # If the directory exists, the text color is black, otherwise it is red
        def check_directory(*args):
            path = self.tx_work_dir.get()
            if path and os.path.isdir(path):
                self.tx_work_dir.config(fg='black')
            else:
                self.tx_work_dir.config(fg='red')
        self.tx_work_dir.bind("<KeyRelease>", check_directory)

        # In the grid below the working directory entry, it is separated into two equal-width columns
        # The first column is for the buttons, the second column is for the text entry
        # The first column is named "col_buttons" and the second column is named "col_text_entry"
        col_buttons = tk.Frame(root, name="col_buttons")
        col_buttons.grid(row=1, column=0, padx=5, pady=5, sticky='ns')
        col_text_entry = tk.Frame(root, name="col_text_entry")
        col_text_entry.grid(row=1, column=1, padx=5, pady=5, sticky='ns')

        # In col_buttons, there is a button, named bt_load, titled "Load ..."
        # bt_load binds with a function: bt_load_clicked
        bt_load = tk.Button(col_buttons, text="Load ...", name="bt_load")
        bt_load.grid(row=0, column=0, padx=5, pady=5, sticky='w')
        ImproToolTip(bt_load, "Load data from a file")
        def bt_load_clicked(*args):
            # Open file dialog to select a file to load
            file_path = filedialog.askopenfilename(title="Select a file to load", filetypes=[("All files", "*.*")])
            if file_path:
                # Here you can add code to handle the loaded file
                self.tx_info_output.insert(tk.END, f"Loaded file: {file_path}\n")
        bt_load.bind("<Button-1>", bt_load_clicked)
        
        # Below bt_load, there is a button, named bt_save, titled "Save ..."
        # bt_save binds with a function: bt_save_clicked
        bt_save = tk.Button(col_buttons, text="Save ...", name="bt_save")
        bt_save.grid(row=1, column=0, padx=5, pady=5, sticky='w')
        ImproToolTip(bt_save, "Save data to a file")
        def bt_save_clicked(*args):
            # Open file dialog to select a file to save
            file_path = filedialog.asksaveasfilename(title="Select a file to save", defaultextension=".txt", filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
            if file_path:
                # Here you can add code to handle the saved file
                self.tx_info_output.insert(tk.END, f"Saved data to: {file_path}\n")
        bt_save.bind("<Button-1>", bt_save_clicked)

        # in col_text_entry
        # tx_info_output: Entry: Information output
        # a text entry for displaying multi-line information
        # This widget is below the working directory entry, at right hand side, multiple lines of height
        self.tx_info_output = tk.Text(col_text_entry, name="tx_infoOutput", height=10, width=50)
        self.tx_info_output.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky='we')
        ImproToolTip(self.tx_info_output, "Information output area\nYou can write multi-line text here.\nUse it to display information or logs.")

        # Text 

        # Configure column weight so Entry expands with window
#        root.grid_columnconfigure(1, weight=1)


if __name__ == "__main__":
    root = tk.Tk()
    app = Mss0(root)
    root.mainloop()
