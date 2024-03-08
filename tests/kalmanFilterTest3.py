# Mouse Tracking with Kalman Filter
# Author: Du Ang
# Material references: https://blog.csdn.net/angelfish91/article/details/61768575
# Date: July 1, 2018

import cv2
import numpy as np


class KalmanFilter:
    """
    Simple Kalman filter
    """

    def __init__(self, X, F, Q, Z, H, R, P, B=np.array([0]), M=np.array([0])):
        """
        Initialise the filter
        Args:
            X: State estimate
            P: Estimate covariance
            F: State transition model
            B: Control matrix
            M: Control vector
            Q: Process noise covariance
            Z: Measurement of the state X
            H: Observation model
            R: Observation noise covariance
        """
        self.X = X
        self.P = P
        self.F = F
        self.B = B
        self.M = M
        self.Q = Q
        self.Z = Z
        self.H = H
        self.R = R

    def predict(self):
        """
        Predict the future state
        Args:
            self.X: State estimate
            self.P: Estimate covariance
            self.B: Control matrix
            self.M: Control vector
        Returns:
            updated self.X
        """
        # Project the state ahead
        self.X = self.F @ self.X + self.B @ self.M
        self.P = self.F @ self.P @ self.F.T + self.Q

        return self.X

    def correct(self, Z):
        """
        Update the Kalman Filter from a measurement
        Args:
            self.X: State estimate
            self.P: Estimate covariance
            Z: State measurement
        Returns:
            updated X
        """
        K = self.P @ self.H.T @ np.linalg.inv(self.H @ self.P @ self.H.T + self.R)
        self.X += K @ (Z - self.H @ self.X)
        self.P = self.P - K @ self.H @ self.P

        return self.X


TITLE = "Mouse Tracking with Kalman Filter"
frame = np.ones((800,800,3),np.uint8) * 255


def mousemove(event, x, y, s, p):
    global frame, current_measurement, current_prediction
    if event == cv2.EVENT_LBUTTONDOWN:
        current_measurement = np.array([[np.float32(x)], [np.float32(y)]])
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
    
        kalman.correct(current_measurement)

    return


cv2.namedWindow(TITLE)
cv2.setMouseCallback(TITLE, mousemove)

stateMatrix = np.zeros((4, 1), np.float32)  # [x, y, delta_x, delta_y]
estimateCovariance = np.eye(stateMatrix.shape[0])
transitionMatrix = np.array([[1, 0, 1, 0],[0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], np.float32) * 0.001
measurementStateMatrix = np.zeros((2, 1), np.float32)
observationMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
measurementNoiseCov = np.array([[1,0],[0,1]], np.float32) * 1
kalman = KalmanFilter(X=stateMatrix,
                      P=estimateCovariance,
                      F=transitionMatrix,
                      Q=processNoiseCov,
                      Z=measurementStateMatrix,
                      H=observationMatrix,
                      R=measurementNoiseCov)

while True:
    cv2.imshow(TITLE,frame)
    ikey = cv2.waitKey(1)
    if ikey & 0xFF == ord('q') or ikey == 27:
        break

cv2.destroyAllWindows()