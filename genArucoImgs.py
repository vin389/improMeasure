# genArucoImgs.py - Generate Aruco images, put them in one image, and returns this image
import os
import cv2
import numpy as np
from colonRangeToIntList import colonRangeToIntList

def genArucoImgs(dict_id=-1, num_markers_x=-1, num_markers_y=-1, 
                 marker_size=-1, gap_size=-1, filename='',
                 displayInfo=True):
    """
    This function generates Aruco images, put them in one image, 
    and returns this image.
    If filename is assigned, also save it to a file.
    Inputs:
        dict_id: int, id of the Aruco dictionary
        num_markers_x: int, number of markers along x in the marker grid
        num_markers_y: int, number of markers along y in the marker grid
        marker_size: int, size of each marker in pixels
        gap_size: int, size of the gap between markers in pixels
        filename: str, filename to save the image
    """
    # display all aruco dictionary names and ask user to choose one
    if dict_id == -1:
        for i in range(99):
            trial_dict = cv2.aruco.getPredefinedDictionary(i)
            markerSize = trial_dict.markerSize
            numMarkers = trial_dict.bytesList.shape[0]
            if i > 0 and markerSize == 4 and numMarkers == 50:
                break
            else:
                print("Dictionary id: Name:   %d: DICT_%dX%d_%d" % (i, markerSize, markerSize, numMarkers))
        # ask user to input an index of dictionary and store it in dict_id
        print("# Please enter a dictionary id from the list above.")
        dict_id = int(input("#  Enter the dictionary id: "))

    # get aruco dictionary of dict_id and store it in aruco_dict
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)

    # print the number of markers in the dictionary
    if displayInfo:
        print("# ArUco dictionary: DICT_%dX%d_%d" % (aruco_dict.markerSize, aruco_dict.markerSize, aruco_dict.bytesList.shape[0]))

    # ask the user the numbers of markers along x and y in the marker grid
    # store the numbers in num_markers_x and num_markers_y
    if num_markers_x == -1 or num_markers_y == -1:
        print("# Please enter the number of markers along x and y in the marker grid.")
        num_markers_x = int(input("#   Enter the number of markers along x: "))
        num_markers_y = int(input("#   Enter the number of markers along y: "))
    if displayInfo:
        print("# Number of markers along x: %d" % num_markers_x)
        print("# Number of markers along y: %d" % num_markers_y)
    
    # ask the user the size of each marker in pixels
    # store the size in marker_size
    if marker_size == -1:
        marker_size = int(input("# Enter the size (# of pixels) of each marker: "))
    if displayInfo:
        print("# Size of each marker: %d" % marker_size)

    # ask the user the size of the gap between markers in pixels
    # store the size in gap_size
    if gap_size == -1:
        gap_size = int(input("# Enter the size (# of pixels) of the gap between markers: "))
    if displayInfo:
        print("# Size of the gap between markers: %d" % gap_size)

    # generate an empty image with the size of the marker grid
    # store the image in img
    num_pixels_x = num_markers_x * (marker_size + gap_size) - gap_size
    num_pixels_y = num_markers_y * (marker_size + gap_size) - gap_size
    img = np.ones((num_pixels_y, num_pixels_x), dtype=np.uint8) * 255
    if displayInfo:
        print("# Image size: %d x %d" % (num_pixels_x, num_pixels_y))

    # generate ArUco markers and put them in img
    for i in range(num_markers_y):
        for j in range(num_markers_x):
            # generate an aruco marker of index i * num_markers_x + j
            marker = cv2.aruco.generateImageMarker(aruco_dict, i * num_markers_x + j, marker_size)
            # put the marker in img
            img[i * (marker_size + gap_size): (i + 1) * (marker_size + gap_size) - gap_size, j * (marker_size + gap_size): (j + 1) * (marker_size + gap_size) - gap_size] = marker

    # display img
    if displayInfo:
        cv2.imshow("Aruco Markers", img)
        cv2.waitKey(0)
        cv2.destroyWindow("Aruco Markers")

    # save img to a file that user assigned. If user assigned a one-character file, then do not save it.
    # if tkinter is available, then use tkinter to ask the user to input the filename from a file dialog
    # if tkinter is not available, then use input() to ask the user to input the filename
    if filename == "":
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
#            filename = filedialog.asksaveasfilename(title="Save the image", filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")])
            filename = filedialog.asksaveasfilename(title="Save the image")
            if filename != "":
                cv2.imwrite(filename, img)
        except:
            filename = input("# Enter the filename to save the image (or enter . to skip it): ")
            if filename != ".":
                cv2.imwrite(filename, img)
    else:
        cv2.imwrite(filename, img)

    # return img
    return img

# a main program that tests genArucoImgs()
if __name__ == "__main__":
    save_directory = r'D:\ExpDataSamples\20240600_CarletonShakeTableCeilingSystem\markers_test_aruco_20240916'
    img = genArucoImgs(dict_id=3, # DICT_4X4_1000
        num_markers_x=1, num_markers_y=2,
        marker_size=200, gap_size=20, 
        filename=os.path.join(save_directory, "dict_3.png"), 
        displayInfo=True)

    img = genArucoImgs(dict_id=11, # DICT_6X6_1000
        num_markers_x=1, num_markers_y=2,
        marker_size=200, gap_size=20, 
        filename=os.path.join(save_directory, "dict_11.png"), 
        displayInfo=True)

    print("# Done.")




