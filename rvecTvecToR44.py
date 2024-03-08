import numpy as np
import cv2 as cv

def rvecTvecToR44(rvec, tvec): # the same as r44FromRvecTvec
    """
    Returns the 4-by-4 coordinate transformation matrix of the camera

    Parameters
    ----------
    rvec: TYPE np.array(3, dtype=float)
    tvec: TYPE np.array(3, dtype=float)
    
    r44 : TYPE np.array((4,4),dtype=float)
        The 4-by-4 form of camera extrinsic parameters
    Returns
    -------
    TYPE: tuple ([0]: np.array(3,dtype=float, [1]: np.array(3,dtype=float)))
    the 4-by-4 coordinate transformation matrix of the camera

    """
    r44 = np.eye(4, dtype=np.float)
    r33, r33joc = cv.Rodrigues(rvec)
    r44[0:3,0:3] = r33.copy()
    r44[0:3,3] = tvec.copy()
    return r44
