# Mouse Tracking with Kalman Filter
# Author: Du Ang
# Material references: https://blog.csdn.net/angelfish91/article/details/61768575
# Date: July 1, 2018

import cv2
import numpy as np

TITLE = "Mouse Tracking with Kalman Filter"
frame = np.ones((800,800,3),np.uint8) * 255


def mousemove(event, x, y, s, p):
    global frame, current_measurement, current_prediction
    if event == cv2.EVENT_LBUTTONDOWN:
#    if event == cv2.EVENT_MOUSEMOVE:
        current_measurement = np.array([[np.float32(x)], [np.float32(y)]])
        kalman.correct(current_measurement)

        current_prediction = kalman.predict()
    
        # cmx, cmy = current_measurement[0], current_measurement[1]
        # cpx, cpy = current_prediction[0], current_prediction[1]
        cmx, cmy = int(current_measurement[0]), int(current_measurement[1])
        cpx, cpy = int(current_prediction[0]), int(current_prediction[1])
        
    
        frame = np.ones((800,800,3),np.uint8) * 255
        cv2.putText(frame, "Measurement: ({:.1f}, {:.1f})".format(np.float32(cmx), np.float32(cmy)),
                    (30, 30), cv2.FONT_HERSHEY_DUPLEX, 0.8, (50, 150, 0))
        cv2.putText(frame, "Prediction: ({:.1f}, {:.1f})".format(np.float32(cpx), np.float32(cpy)),
                    (30, 60), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255))
        if cmx > 0 and cmx < 800 and cmy > 0 and cmy < 800:
            cv2.circle(frame, (cmx, cmy), 10, (50, 150, 0), -1)      # current measured point
        if cpx > 0 and cpx < 800 and cpy > 0 and cpy < 800:
            cv2.circle(frame, (cpx, cpy), 10, (0, 0, 255), -1)      # current predicted point
    
#        kalman.correct(current_measurement)

    return


cv2.namedWindow(TITLE)
cv2.setMouseCallback(TITLE, mousemove)

kalman = cv2.KalmanFilter(4, 2, 0)
kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
kalman.processNoiseCov = np.float32(1e-2) * np.eye(4, dtype=np.float32)
kalman.measurementNoiseCov = np.float32(0.1) * np.ones((2, 2), dtype=np.float32)
kalman.errorCovPost = np.float32(1) * np.ones((4, 4), dtype=np.float32)
#kalman.measurementMatrix = np.ones((1, 4), dtype=np.float32)
kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)

while True:
    cv2.imshow(TITLE,frame)
    ikey = cv2.waitKey(1)
    if ikey & 0xFF == ord('q') or ikey == 27:
        break

cv2.destroyAllWindows()