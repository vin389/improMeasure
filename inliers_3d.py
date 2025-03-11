import numpy as np
from scipy.spatial import distance

def detect_outliers_knn(points, k=5, percentile=95):
    """
    Detects outliers in a 3D dataset using k-Nearest Neighbors (k-NN).
    
    Parameters:
        points (ndarray): N × 3 array of points in 3D space.
        k (int): Number of nearest neighbors to consider.
        percentile (float): Percentile threshold to classify outliers (default: 95).
        
    Returns:
        outliers (ndarray): Boolean array where True indicates an outlier.
    """
    N = len(points)
    dist_matrix = distance.cdist(points, points, metric='euclidean')  # Compute pairwise distances
    # Sort distances and take the k-nearest (excluding self)
    nearest_distances = np.sort(dist_matrix, axis=1)[:, 1:k+1]  # Exclude the self-distance (0)
    # Compute the mean distance to k-nearest neighbors
    mean_distances = np.mean(nearest_distances, axis=1)
    # Define the outlier threshold using the 95th percentile of distances
    threshold = np.percentile(mean_distances, percentile)
    # Points with mean distances greater than threshold are outliers
    outliers = mean_distances > threshold
    return outliers

# This function calculates the centroid of the inliers in a 3D dataset.
# Input:
#   x3d: N x 3 array of points in 3D space. N is the number of points.
#   k: Number of nearest neighbors to consider.
#   percentile: Percentile threshold to classify outliers.
# Output:
#   centroid: 3D coordinates of the centroid of inliers.
def inliers_centroid_3d(x3d, k=5, percentile=95):
    """
    Calculates the centroid of the inliers in a 3D dataset.
    Parameters:
        x3d (ndarray): N × 3 array of points in 3D space. N is the number of points.
    Returns:
        centroid (ndarray): 3D coordinates of the centroid of inliers.
    """
    # Detect outliers using k-NN
    outliers = detect_outliers_knn(x3d, k=5, percentile=80)
    # Compute the centroid of inliers
    inliers = x3d[~outliers, :]
    centroid = np.mean(inliers, axis=0)
    return centroid



# Test the function find_inliers_3d
if __name__ == '__main__':
    # Number of points
    N = 100

    # Generate random 3D points
    x3d = np.zeros((N, 3), dtype=np.float32)
    x3d[:, 0] = 1 
    x3d[:, 1] = 2 
    x3d[:, 2] = 3 
    # Add some noise
    x3d = x3d + 0.05 * np.random.randn(N, 3)
    # Add some outliers
    x3d[2:5, :] = 10 * np.random.rand(3, 3)
    print(x3d)

    # Calculate the centroid of inliers
    centroid = inliers_centroid_3d(x3d, k=5, percentile=80)
    print('Centroid of inliers:', centroid)
