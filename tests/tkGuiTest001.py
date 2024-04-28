import tkinter as tk

def copy_text():
  # Get the text from the first entry box
  text_to_copy = entry1.get(0., tk.END)
  # Clear the second entry box
  entry2.delete("1.0", tk.END)
  # Insert the copied text into the second entry box
  entry2.insert(tk.END, text_to_copy)

# Create the main window
window = tk.Tk()
window.title("Text Copier")

# Create the first entry box with wider width
entry1 = tk.Text(window, width=50, height=2, undo=True, maxundo=-1)
entry1.grid(row=0, column=0, padx=10, pady=10)

# Create the button
copy_button = tk.Button(window, text="Copy Text", command=copy_text)
copy_button.grid(row=2, column=0, padx=10, pady=10)

# Create the second entry box
entry2 = tk.Text(window, width=50, height=2)
entry2.grid(row=4, column=0, padx=10, pady=10)

# Create the second entry box
entry3 = tk.Text(window, width=50, height=2)
entry3.grid(row=4, column=1, padx=10, pady=10)


# Start the main event loop
window.mainloop()