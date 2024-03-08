import numpy as np
import cv2
import matplotlib.pyplot as plt

# Generate some sample data
np.random.seed(0)
n_samples = 100
x = np.linspace(0, 10, n_samples)
y = 0.5 * x + np.random.normal(0, 1, n_samples)

# Create a Kalman filter
kalman = cv2.KalmanFilter(2, 1, 0)

# Initialize the Kalman filter
kalman.transitionMatrix = np.array([[1., 1.], [0., 1.]])
kalman.measurementMatrix = 1. * np.ones((1, 2))
kalman.processNoiseCov = 1e-5 * np.eye(2)
kalman.measurementNoiseCov = 1e-1 * np.ones((1, 1))
kalman.errorCovPost = 1. * np.ones((2, 2))
kalman.statePost = 0.1 * np.random.randn(2, 1)

# Run the Kalman filter on the data
filtered_state_means = []
for i in range(len(x)):
    kalman.correct(np.array([[y[i]]]))  # Wrap y[i] in a 2D array
    filtered_state_means.append(kalman.statePost[0])

# Plot the original data and the filtered result
plt.plot(x, y, label='Original Data')
plt.plot(x, filtered_state_means, label='Filtered Result')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Kalman Filter Example')
plt.show()