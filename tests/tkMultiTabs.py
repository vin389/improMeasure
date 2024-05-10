import tkinter as tk
from tkinter import ttk  # for notebook (tabs)


root = tk.Tk()
root.title("Two Tab GUI")

# Create the notebook (tabs)
notebook = ttk.Notebook(root)
notebook.pack(expand=True, fill="both")

# Create the first tab
tab1 = ttk.Frame(notebook)
notebook.add(tab1, text="Tab 1")

# Create the second tab
tab2 = ttk.Frame(notebook)
notebook.add(tab2, text="Tab 2")

# Add content to each tab
label1 = ttk.Label(tab1, text="This is the content of the first tab.")
label1.pack()
label2 = ttk.Label(tab2, text="This is the content of the second tab.")
label2.pack()


root.mainloop()