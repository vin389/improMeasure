import numpy as np

def permuteRows(thePoints: np.ndarray, theIdx: np.ndarray):
    """
    This function generates a 2-d array that is reordered.
    For example, 
        thePoints = np.array([[3., 4., 5.],
                              [4., 5., 7.],
                              [4., 3., 2.],
                              [5., 3., 5.],
                              [4., 3., 2.]])
        theIdx = np.array([-1,  0, -1,  1, -1,  2,  3,  4])
        theOrderedPoints = permuteRows(thePoints, theIdx)
        theOrderedPoints would be 
        np.array([[nan, nan, nan],
                  [ 3.,  4.,  5.],
                  [nan, nan, nan],
                  [ 4.,  5.,  7.],
                  [nan, nan, nan],
                  [ 4.,  3.,  2.],
                  [ 5.,  3.,  5.],
                  [ 4.,  3.,  2.]])
        If theIdx are all non-negative numbers, the statement can be
        directly replaced by 
        theOrderedPoints = thePoints[theIdx]

    Parameters
    ----------
    thePoints : np.ndarray
        DESCRIPTION.
    theIdx : np.ndarray
        DESCRIPTION.

    Returns
    -------
    None.

    """
    theIdx = theIdx.flatten()
    nRows = theIdx.size
    nCols = thePoints.shape[1]
    if thePoints.dtype==float or thePoints.dtype==np.float32 or \
       thePoints.dtype==np.float64 or thePoints.dtype==np.float64:
        theArray = np.ones((nRows, nCols), dtype=thePoints.dtype) * np.nan
    else:
        theArray = np.ones((nRows, nCols), dtype=thePoints.dtype) * (-1)
        
    for i in range(nRows):
        if theIdx[i] >= 0:
            theArray[i] = thePoints[theIdx[i]]
    return theArray

    
    
