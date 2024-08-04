import numpy as np
import cv2 as cv
import os, sys, glob, time, re
from inputs import input2

# Function videoFramesToPicsAndVideo
# This function takes a video file, indices of frames to extract, and a directory to save the frames to.
# It extracts the frames and saves them as images in the directory.
# It also creates a video from the extracted frames and saves it in the same directory.
def videoFramesToPicsAndVideo(videoPath="", frameIndices=[], saveDir="", frameIdxShift=-1):
    # if the videoPath does not exist, ask the user to input the video path in terminal
    if not os.path.exists(videoPath):
        videoPath = input2("# Enter the video path:\n")
    # if the videoPath does not exist, ask the user to input the video path again.
    if not os.path.exists(videoPath):
        # ask the user to input again
        print("# The video path does not exist. Please enter a valid video path.")
        videoPath = input2("# Enter the video path:\n")
    # if the videoPath still does not exist, print an error message and return
    if not os.path.exists(videoPath):
        print("# The video path does not exist.")
        return
    # open the video file
    cap = cv.VideoCapture(videoPath)
    # if the video file cannot be opened, print an error message and return
    if not cap.isOpened():
        print("# Error: Cannot open video file.")
        return
    
    # if the frameIndices is empty, ask user to input the frame indices in terminal. 
    # allowing the user to input multiple frame indices separated by commas or spaces.
    while len(frameIndices) == 0:
        frameIndices = input2("# Enter the frame indices (1-base) (separated by commas or spaces) (any negative value to quit):\n")
        # split the input string by commas or spaces and convert the substrings to integers
        frameIndices = [int(i) for i in re.split(r',|\s', frameIndices) if i]
        # if any element is negative, print a warning message and return
        if any([i < 0 for i in frameIndices]):
            print("# Frame indices has negative value. Quit.")
            return

    # if the saveDir is empty, ask the user to input the save directory in terminal
    while saveDir == "":
        saveDir = input2("# Enter the save directory:\n").strip()

#    # if the saveDir does not exist, ask the user to input the save directory in terminal
#    while not os.path.exists(saveDir):
#        print("# The save directory does not exist.")
#        saveDir = input2("# Enter the save directory:\n").strip()

    # create the save directory if it does not exist
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)

    # if the frameIdxShift is negative, ask the user to input the frame index shift in terminal
    if frameIdxShift < 0:
        frameIdxShift = int(input2("# Enter the frame index shift for file name (default is 0, meaning the file name frame index is the index):\n"))

    # get the frame count of the video
    frameCount = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    # get the frame width and height of the video
    frameWidth = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    # get the frame rate of the video
    frameRate = cap.get(cv.CAP_PROP_FPS)

    # create a new video file to save the extracted frames with the same resolution and frame rate as the original video
    videoOut = cv.VideoWriter(os.path.join(saveDir, "output_%d.mp4" % frameIdxShift), cv.VideoWriter_fourcc(*'mp4v'), frameRate, (frameWidth, frameHeight))

    useSet = False
    if useSet:
        # loop through the frameIndices and save the frames as images. Use opencv set function to set the frame position
        # and read function to read the frame. Also add the frame into a new video file.
        # This method is faster than reading frame by frame.
        for frameIndex in frameIndices:
            cap.set(cv.CAP_PROP_POS_FRAMES, frameIndex)
            ret, frame = cap.read()
            if ret:
                frameIndexForFile = frameIndex + frameIdxShift
                cv.imwrite(os.path.join(saveDir, f"frame_{frameIndexForFile:06d}.jpg"), frame)
                videoOut.write(frame)
        # release the video file
        cap.release()
        # release the video writer
        videoOut.release()
    else: # use frame by frame reading
        # loop through the video frame by frame. If the frame is in frameIndices, save it as an image and add it to the video.
        for i in range(1, frameCount+1):
            if i % 100 == 0:
                # print the frame information 
                print('\b'*100, end='')
                print(f"# Processing frame {i}/{frameCount}", end='')
                # flush the print buffer
                sys.stdout.flush()
            # read the frame
            ret, frame = cap.read()
            if i in frameIndices:
                frameIndexForFile = i + frameIdxShift
                cv.imwrite(os.path.join(saveDir, f"frame_{frameIndexForFile:06d}.jpg"), frame)
                videoOut.write(frame)
                # print the saved image file name
                print(f"Saved %s" % os.path.join(saveDir, f"frame_{frameIndexForFile:06d}.jpg"))
            # if all the frames in frameIndices have been processed, break the loop
            if i > max(frameIndices):
                break
        # release the video file
        cap.release()
        # release the video writer
        videoOut.release()

# An example to demonstrate how to use this function
if __name__ == '__main__':
    videoFramesToPicsAndVideo()

