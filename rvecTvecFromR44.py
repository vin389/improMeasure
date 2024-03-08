import cv2 as cv

def rvecTvecFromR44(r44): 
    """
    Returns the rvec and tvec of the camera

    Parameters
    ----------
    r44 : TYPE np.array, (4,4), dtype=float)
        The 4-by-4 form of camera extrinsic parameters
    Returns
    -------
    TYPE: tuple (np.array((3,1),dtype=float)
    Returns the rvec and tvec of the camera ([0] is rvec; [1] is tvec.)

    """
    rvec, rvecjoc = cv.Rodrigues(r44[0:3,0:3])
    tvec = r44[0:3,3]
    return rvec.reshape(3,1), tvec.reshape(3,1)
