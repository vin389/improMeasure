# import opencv and numpy
import cv2
import numpy as np
import time

# create a VideoCapture object in opencv 
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

# check if the camera is opened
if not cap.isOpened():
    print("Cannot open camera")
    exit()

# if camera is opened, switch the resolution to 640x480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840) # 4k/high_res
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160) # 4k/high_res
par_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
par_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
par_fps = cap.get(cv2.CAP_PROP_FPS)
print('# resolution %dx%d.FPS:%d' % (par_width, par_height, par_fps))

# loop to read the frames from the camera
fps_count = 0
count_fps_per = 30
t1 = time.time()
fps_txt = ''
toImshow = True
while True:
    # read the frame from the camera
    ret, frame = cap.read()

    # get the size of the image
    height, width, channels = frame.shape

    # put text of width and height on the upper-left corner of the image
    if toImshow:
        cv2.putText(frame, 'Width: '+str(width), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
        cv2.putText(frame, 'Height: '+str(height), (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

    # calculate the frames per second
    fps_count += 1
    if fps_count % count_fps_per == 0:
        t2 = time.time()
        fps = count_fps_per / (t2-t1)
        t1 = t2
        fps_txt = 'FPS: %.2f' % fps

    # if toImshow is True, put the fps on the upper-left corner of the image
    if toImshow == True:
        cv2.putText(frame, fps_txt, (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
    else:
        print('\b'*20 + fps_txt, end='')

    # if the image is wider than 640 pixels, resize it to 640 pixels
    if toImshow:
        winMaxWidth = 1400
        if width > winMaxWidth:
            frame = cv2.resize(frame, (winMaxWidth, int(winMaxWidth*height/width)))

    # check if the frame is read correctly
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # display the frame
    if toImshow:
        cv2.imshow('frame', frame)

    # check if the user pressed the 'q' key
    if cv2.waitKey(1) == ord('q'):
        print()
        break
