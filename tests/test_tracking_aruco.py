import cv2
import numpy as np

def track_aruco(cap):
    # get aruco dictionary
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
    

    aruco_dict = cv2.aruco.Dictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()

    while True:
        ret, frame = cap.read()

        # Detect Aruco markers in the frame
        corners, ids, rejected = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=aruco_params)

        # If at least one marker was detected
        if len(corners) > 0:
            # Estimate pose of the marker
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarker(corners[0], aruco_dict, 1, np.zeros((3, 1)), np.zeros((3, 3)))

            # Draw the detected marker and its pose
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            cv2.drawFrameAxes(frame, rvec, tvec, 100)

        # Display the frame
        cv2.imshow('Aruco Tracking', frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Create a VideoCapture object to capture from the webcam
cap = cv2.VideoCapture(0)

# Start tracking the Aruco marker
track_aruco(cap)

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()