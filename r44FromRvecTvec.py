import numpy as np
import cv2 as cv


def r44FromRvecTvec(rvec, tvec): 
    """
    Returns the 4-by-4 coordinate transformation matrix of the camera

    Parameters
    ----------
    rvec: TYPE np.array, shape is (3,), (3,1), or (1,3), dtype=float32 or 64
        the rotational vector 
    tvec: TYPE np.array, shape is (3,), (3,1), or (1,3), dtype=float32 or 64
        the translational vector

    Returns
    -------
    TYPE np.array((4,4),dtype=float)
         The 4-by-4 form of camera extrinsic parameters (np.float64)
    """
    # data check
    if type(rvec) != np.ndarray or rvec.size != 3 or rvec.dtype != np.float64:
        rvec = np.array(rvec, dtype=np.float64).reshape(3, 1)
    if type(tvec) != np.ndarray or tvec.shape != (3,1) or tvec.dtype != np.float64:
        tvec = np.array(tvec, dtype=np.float64).reshape(3, 1)
    # return
    return _r44FromRvecTvec(rvec, tvec)


def _r44FromRvecTvec(rvec, tvec): 
    """
    Returns the 4-by-4 coordinate transformation matrix of the camera
    without checking inputs

    Parameters
    ----------
    rvec: TYPE np.array, shape is (3,), (3,1), or (1,3), dtype=float
    tvec: TYPE np.array, shape is (3,1), dtype=float
    
    r44 : TYPE np.array((4,4),dtype=float)
        The 4-by-4 form of camera extrinsic parameters
    Returns
    -------
    TYPE np.array((4,4),dtype=float)
         The 4-by-4 form of camera extrinsic parameters (np.float64)
    """
    r44 = np.eye(4, dtype=np.float64)
    r44[0:3,0:3] = cv.Rodrigues(rvec)[0]
    r44[0:3,3:4] = tvec
    return r44
