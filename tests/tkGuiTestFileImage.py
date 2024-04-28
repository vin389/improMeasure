import tkinter as tk
from PIL import Image, ImageTk
import cv2

def open_jpg():
  """Opens a file dialog, validates JPG selection, and displays the image."""
  filename = tk.filedialog.askopenfilename(filetypes=[("JPG Images", "*.jpg")])
  if filename:
    try:
      # Open the image with PIL (Pillow Fork)
#      image = Image.open(filename)
#      image = image.resize((300, 300), Image.LANCZOS)  # Resize for display
#      photo = ImageTk.PhotoImage(image)
      # Open the image with cv2
      image = cv2.imread(filename)
      image = cv2.resize(image, dsize=(0,0), fx=0.1, fy=0.1)
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      photo = ImageTk.PhotoImage(image=Image.fromarray(image))
      # Update the label with the resized image
      image_label.config(image=photo)
      image_label.image = photo  # Keep a reference to avoid garbage collection
    except (IOError, OSError) as e:
      # Handle potential errors like invalid file or format
      print(f"Error opening file: {e}")

# Create the main window
window = tk.Tk()
window.title("Image Viewer")

# Create a button to open the file dialog
open_button = tk.Button(window, width=50, text="Open JPG", command=open_jpg)
open_button.grid(row=0, column=0, padx=10, pady=10)

# Create a label to display the image
image_label = tk.Label(window)
image_label.grid(row=1, column=0, padx=10, pady=10)

# Start the main event loop
window.mainloop()
