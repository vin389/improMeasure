def cel2fah(cel):
    return (cel * 9. / 5.) + 32.
def fah2cel(fah):
    return (fah - 32.) * 5. / 9.

import tkinter as tk
# Create widgets
root = tk.Tk()
root.title("Temperature Converter")
label_cel = tk.Label(root, text="Cel:")
label_fah = tk.Label(root, text="Fah:")
text_cel = tk.Text(root, width=30, height=2)
text_fah = tk.Text(root, width=30, height=2)
# Layout widgets
label_cel.grid(row=0, column=0)
text_cel.grid(row=0, column=1)
label_fah.grid(row=1, column=0)
text_fah.grid(row=1, column=1)

# Define commands
def cmd_cel(event):
  try:
    cel = float(text_cel.get(1.0, tk.END))
    fah = cel2fah(cel)
    text_fah.delete(0., tk.END); text_fah.insert(0., f"{fah:.2f}")
  except ValueError:
    text_cel.delete(0., tk.END); text_fah.delete(0., tk.END)

def cmd_fah(event):
  try:
    fah = float(text_fah.get(1.0, tk.END))
    cel = fah2cel(fah)
    text_cel.delete(0., tk.END); text_cel.insert(0., f"{cel:.2f}")
  except ValueError:
    text_cel.delete(0., tk.END); text_fah.delete(0., tk.END)

# Set commands
text_cel.bind("<KeyRelease>", cmd_cel)  
text_fah.bind("<KeyRelease>", cmd_fah)

# Run loop
root.mainloop()