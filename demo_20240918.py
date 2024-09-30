import tkinter as tk
from tkinter import Label
import cv2
import numpy as np

def convert_image_to_tkinter(img):
    # Encode OpenCV image as a PNG in memory
    _, buffer = cv2.imencode('.png', img)
    # Convert buffer to byte array and decode as a Tkinter image
    return tk.PhotoImage(data=buffer.tobytes())

# Function to handle "Start" button click
def start_action():
    # create a cv2 VideoCapture
    while True:
        cam1 = cv2.VideoCapture(0)
        if cam1.isOpened() == True:
            break
        else:
            print("# Error. This proram cannot open web cam (VideoCapture 0)")
            print("# This program will wait for 1 second and try again.")
            cv2.waitKey(1000)

    while True:
        img_check, img = cam1.read()

        # Convert to a Tkinter-compatible format
        img_tk = convert_image_to_tkinter(img)

        # display image 
        # Convert to a Tkinter-compatible format
        img_tk = convert_image_to_tkinter(img)
        # Display the image in the label
        image_label.config(image=img_tk)
        image_label.image = img_tk  # Keep a reference to avoid garbage collection
        # detect key
        ikey = cv2.waitKey(1)
        if ikey == 32 or ikey == 27 or ikey == 'q':
            break
#        cv2.imshow("demo", img)
#        ikey = cv2.waitKey(1)
#        if ikey == 32 or ikey == 27 or ikey == 'q':
#            cv2.destroyWindow("demo")
#            break
   
    # close video
    cam1.release()

def end_action():
    # Clear the image
    image_label.config(image='')
    image_label.image = None

# create a main program
if __name__ == '__main__':

    # Create the main window
    window = tk.Tk()
    window.title("Image Display Window")
    window.geometry("500x400")

    # Create buttons
    start_button = tk.Button(window, text="Start", command=start_action)
    end_button = tk.Button(window, text="End", command=end_action)

    # Pack buttons to the left side of the window
    start_button.pack(side=tk.LEFT, padx=10, pady=10)
    end_button.pack(side=tk.LEFT, padx=10, pady=10)

    # Create a label to display the image
    image_label = Label(window)
    image_label.pack(side=tk.RIGHT, padx=10, pady=10, expand=True)

    # Start the Tkinter event loop
    window.mainloop()


