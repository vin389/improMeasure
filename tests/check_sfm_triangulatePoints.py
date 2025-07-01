import cv2
import numpy as np

# Sample 2D points (replace with your actual data)
points2d = [
    np.array([[100, 200], [150, 250], [200, 300]], dtype=np.float32).T,  # Image 1
    np.array([[110, 210], [160, 260], [210, 310]], dtype=np.float32).T   # Image 2
]

# Sample projection matrices (replace with your actual matrices)
projection_matrices = [
    np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float32),  # Camera 1
    np.array([[1, 0, 0, 0.1], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float32) # Camera 2
]

# Triangulate points
points3d = cv2.sfm.triangulatePoints(points2d, projection_matrices)

print(points3d)