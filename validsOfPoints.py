import numpy as np

# This file defines the following function(s):
# validsOfPoints
# validsOfPoints2


def validsOfPoints(points: np.ndarray):
    """
    This function allows user to input an array of points which some of them 
    have nan values, and returns a vector that indicates which are valid points.
    For example, 
        points = np.array([
               [ 1.,  2., np.nan],
               [ 3.,  4.,  5.],
               [np.nan,  6., np.nan],
               [ 4.,  5.,  7.],
               [ 3., np.nan,  6.],
               [ 4.,  3.,  2.],
               [ 5.,  3.,  5.],
               [ 4.,  3.,  2.]])
        valids, idx_old2new, idx_new2old = validsOfPoints(points)
    validsOfPoints(x) returns a tuple: 
        [0]: array([ 0, 1,  0, 1,  0, 1, 1, 1], dtype=uint8) (valids)
        [1]: array([-1, 0, -1, 1, -1, 2, 3, 4], dtype=int)  (idx_old2new)
             For example, idx_old2new[0] is -1, indicates that the 
             index 0 in old array does not exist in new array.
             idx_old2new[3] is 1 indicates the index 3 in old array is index 
             1 in the new array. 
        [2]: array([    1,     3,     5, 6, 7], dtype=int)  (idx_new2old)
        
    You can use points[idx_new2old] to obtain the new array:
        array([[3., 4., 5.],
               [4., 5., 7.],
               [4., 3., 2.],
               [5., 3., 5.],
               [4., 3., 2.]])

    Parameters
    ----------
    points : np.ndarray (typcailly n-by-2 or n-by-3, n is number of points)

    Returns
    -------
    valids : TYPE
        DESCRIPTION.
    """
    # initialization
    nValid = 0
    valids = np.ones(points.shape[0], dtype=np.uint8)
    idx_old2new = np.ones(points.shape[0], dtype=int)
    # check nan points by going through the array points
    # create valids[] and idx_new2old[]
    for i in range(points.shape[0]):
        # check if it is an nan point
        for j in range(points.shape[1]):
            if np.isnan(points[i, j]):
                valids[i] = 0
                break
        # set idx_new2old
        if valids[i] == 1:
            idx_old2new[i] = nValid
            nValid += 1
        else:
            idx_old2new[i] = -1
    # create idx_new2old[] from idx_old2new[]
    idx_new2old = np.zeros(nValid, dtype=int)
    for i in range(points.shape[0]):
        if idx_old2new[i] >= 0:
            idx_new2old[idx_old2new[i]] = i
    # return valids, idx_new2old, and idx_old2new
    return (valids, idx_old2new, idx_new2old)


def validsOfPoints2(points1: np.ndarray, points2: np.ndarray):
    """
    This function allows user to input two arrays of points which some of them 
    have nan values, and returns a vector that indicates which are valid points.
    This is designed for camera calibration. The points1 and points2 can be 
    objPoints and imgPoints, respectively. 
    
    For example, 
        points1 = np.array([
               [ 1.,  2., np.nan],
               [ 3.,  4.,  5.],
               [np.nan,  6., np.nan],
               [ 4.,  5.,  7.],
               [ 3., np.nan,  6.],
               [ 4.,  3.,  2.],
               [ 5.,  3.,  5.],
               [ 4.,  3.,  2.]])
        points2 = np.array([
               [ 12.34, 34.45 ], 
               [ 12.34, 34.45 ], 
               [ 12.34, 34.45 ], 
               [ 12.34, 34.45 ], 
               [ 12.34, 34.45 ], 
               [ 12.34, 34.45 ], 
               [ np.nan, np.nan ], 
               [ 12.34, 34.45 ]])
        valids, idx_old2new, idx_new2old = validsOfPoints2(points1, points2)
        The validsOfPoints2(points1, points2) returns a tuple of:
        [0]: np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=uint8)
        [1]: np.array([-1,  0, -1,  1, -1,  2, -1,  3])
        [2]: np.array([1, 3, 5, 7])
        
    You can use points1[idx_new2old] to obtain the new objPoints array:
        array([[3., 4., 5.],
               [4., 5., 7.],
               [4., 3., 2.],
               [5., 3., 5.],
               [4., 3., 2.]])

    Parameters
    ----------
    points1 : np.ndarray
        DESCRIPTION.
    points2 : np.ndarray
        DESCRIPTION.

    Returns
    -------
    valids : TYPE
        DESCRIPTION.
    idx_o2n : TYPE
        DESCRIPTION.
    idx_n2o : TYPE
        DESCRIPTION.

    """
    # check 
    nPoints1 = points1.shape[0]
    nPoints2 = points2.shape[0]
    if nPoints1 != nPoints2:
        print("# Error: validsOfPoints2: points1 and points2 should have the same number of points (rows)")
        print("#        points1.shape: ", points1.shape)
        print("#        points2.shape: ", points2.shape)
        print("#        Instead of returning valids, idx_old2new, and idx_new2old, it is returning None. Sorry.")
        return None
    # Check valids separately
    valids1, idx_o2n_1, idx_n2o_1 = validsOfPoints(points1)
    valids2, idx_o2n_2, idx_n2o_2 = validsOfPoints(points2)
    # Combine valids1 and valids2
    valids = valids1
    for i in range(nPoints1):
        if valids2[i] == 0:
            valids[i] = 0
    # calculate nPointsValid (number of valid points) and idx_o2n
    nPointsValid = 0
    idx_o2n = np.ones(idx_o2n_1.shape, idx_o2n_1.dtype) * (-1)
    for i in range(nPoints1):
        if valids[i] != 0:
            idx_o2n[i] = nPointsValid
            nPointsValid += 1
    # indices mapping all points (0:nPointsAll) to valid points
    idx_n2o = np.ones((nPointsValid), dtype=int) * (-1)
    for i in range(nPoints1):
        if idx_o2n[i] >= 0:
            idx_n2o[idx_o2n[i]] = i
    return (valids, idx_o2n, idx_n2o)