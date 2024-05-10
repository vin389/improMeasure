from tkinter import Tk, ttk, filedialog

# Window size
window_width = 1600
window_height = 900

# Create main window
window = Tk()
window.geometry(f"{window_width}x{window_height}")

# Create notebook (tabs)
notebook = ttk.Notebook(window)
notebook.pack(fill="both", expand=True)

# Create tabs
tab1 = ttk.Frame(notebook)
tab2 = ttk.Frame(notebook)
notebook.add(tab1, text="Tab 1 - Select Files")
notebook.add(tab2, text="Tab 2")

# Button for file selection (Tab 1)
def select_files():
    filenames = filedialog.askopenfilenames(title="Select Files", 
                                           filetypes=[("All Files", "*.*")] )
    selected_files_label.config(text="Selected files: " + ", ".join(filenames))

select_button = ttk.Button(tab1, text="Select Files", command=select_files)
select_button.pack(pady=20)

# Display selected files (initially empty)
selected_files_label = ttk.Label(tab1, text="Selected files: ")
selected_files_label.pack()


# Main loop
window.mainloop()
