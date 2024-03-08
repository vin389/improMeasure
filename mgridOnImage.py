import numpy as np

def mgridOnImage(imgHeight, imgWidth, nHeight, nWidth, 
                 x0=0, y0=0, 
                 dtype=np.float32):
    """
    This function returns 2D points of grid that are uniformly distributed
    over an image. 
    The return array is nHeight by nWidth by 2. 
    This function is designed for optical flow calculation. Before using it
    for optical flow calculation, you need to reshape the array to a N-by-2
    array (from nHeight by nWidth by 2, e.g., 
      prevPts = mgridOnImage(1080, 1920, 90, 160, np.float32).reshape(-1,2)

    Parameters
    ----------
    imgHeight : int (can be float)
        image height (pixel).
    imgWidth : int (can be float)
        image width (pixel).
    nHeight : int
        # of points of this grid along height
    nWidth : int
        # of points of this grid along 
    x0 : int (can be float)
        the x (pixel) of the upper-left corner of the ROI. 
        Default is 0.
    y0 : int (can be float)
        the y (pixel) of the upper-left corner of the ROI. 
        Default is 0.
    dtype : TYPE, optional
        np.float32 or np.float64 (float)

    Returns
    -------
    coordinates of the grid (np.array((nHeight, nWidth, 2)), dtype=dtype)

    """
    dy = imgHeight / nHeight
    dx = imgWidth / nWidth
    pts = np.zeros((nHeight, nWidth, 2), dtype=dtype)
    pts[:,:,1], pts[:,:,0] =\
        np.mgrid[(-0.5*(1-dy) + y0):(imgHeight + y0):dy, 
                 (-0.5*(1-dx) + x0):(imgWidth + x0):dx]
    return pts