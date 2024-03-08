import numpy as np
import cv2 as cv

def rvecTvecFromPosAim(pos, aim):
    """
    This function returns camera extrinsic parameters (rvec and tvec) given
    camera position (pos) and its aiming point (aim). The calculated x-axis of 
    the camera would be on the X-Y plane of the world coordinate.     

    Parameters
    ----------
    pos : np.array, tuple, or list
        coordinate of the position of the camera.
    aim : TYPE
        coordinate of a point that this camera is aiming. Or, a point on the 
        camera's z-axis

    Returns
    -------
    rvec : np.array (3 by 1)
        Rotational vector of the extrinsic parameters
    tvec : np.array (3 by 1)
        Translational vector of the extrinsic parameters
        
    Example
    -------
        from camposFromRvecTvec import camposFromRvecTvec
        pos = [  0, -100, 100]
        aim = [100,    0, 600]
        rvec, tvec = rvecTvecFromPosAim(pos, aim)
        print(camposFromRvecTvec(rvec, tvec))
        # You will get the rvec and tvec, and if you calculate back 
        # the camera position by giving rvec and tvec, you will get
        # [0, -100, 100 ]
    """
    pos = np.array(pos, dtype=float).flatten()
    aim = np.array(aim, dtype=float).flatten()
    zvec = aim - pos
    if zvec[0] == 0. and zvec[1] == 0.:
        # if zvec is vertical upward, xvec has no unique answer, and needs to be preset
        xvec = np.array([0,0,1], dtype=float).flatten()
    else:
        xvec = np.cross(zvec.flatten(), np.array([0,0,1.]))
    yvec = np.cross(zvec, xvec)
    xvec = xvec.reshape(3, 1) / np.linalg.norm(xvec)
    yvec = yvec.reshape(3, 1) / np.linalg.norm(yvec)
    zvec = zvec.reshape(3, 1) / np.linalg.norm(zvec)
    r33inv = np.hstack((xvec, yvec, zvec))
    r44inv = np.eye(4, dtype=float)
    r44inv[0:3, 0:3] = r33inv.copy()
    r44inv[0:3, 3] = pos[:].copy()
    r44 = np.linalg.inv(r44inv)
    rvec = cv.Rodrigues(r44[0:3, 0:3])[0]
    tvec = r44[0:3, 3].reshape((3, 1))
    return rvec.reshape(3, 1), tvec.reshape(3, 1)
