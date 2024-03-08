import numpy as np
import matplotlib.pyplot as plt

class KalmanFilter:
    def __init__(self, A, B, C, D, Q, R):
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.Q = Q
        self.R = R

        # Initialize the state and covariance matrices
        self.state = np.zeros(len(A[:, 0]))
        self.P = np.eye(len(A))

    def predict(self):
        # Predict the next state
        self.state = self.A * self.state + self.B

        # Propagate the covariance matrix
        self.P = self.A * self.P * self.A.T + self.Q

    def update(self, measurement):
        # Compute the Kalman gain
        K = self.P * self.C.T * np.linalg.inv(self.C * self.P * self.C.T + self.R)

        # Update the state estimate
        self.state = self.state + K * (measurement - self.C * self.state)

        # Update the covariance matrix
        self.P = (np.eye(len(self.A)) - K * self.C) * self.P

# Define the system matrices
A = np.array([[1, 0], [0, 1]])
B = np.array([[0], [0]])
C = np.array([[1, 0]])
D = np.array([[0]])

# Define the process and measurement noise covariance matrices
Q = np.array([[0.01, 0], [0, 0.01]])
R = np.array([[0.1]])

# Create a Kalman filter object
kf = KalmanFilter(A, B, C, D, Q, R)

# Generate some noisy measurements
measurements = np.array([[1, 1], [1.1, 1.1], [1.2, 1.2], [1.3, 1.3], [1.4, 1.4]]) + np.random.randn(5, 2)

# Initialize the Kalman filter
kf.state = measurements[0]

# Run the Kalman filter
filtered_measurements = []
for measurement in measurements:
    kf.predict()
    kf.update(measurement)
    filtered_measurements.append(kf.state)

# Plot the results
plt.plot(measurements[:, 0], measurements[:, 1], 'o', label='Measurements')
plt.plot(np.array(filtered_measurements)[:, 0], np.array(filtered_measurements)[:, 1], '-', label='Filtered measurements')
plt.legend()
plt.show()