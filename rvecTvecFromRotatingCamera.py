import numpy as np
import cv2 as cv

def r44Of(rvec, tvec):
    r44 = np.eye(4, dtype=float)
    r44[0:3, 0:3] = cv.Rodrigues(rvec)[0]
    r44[0:3, 3] = tvec.flatten()
    return r44

def r44invOf(rvec, tvec):
    r44 = np.eye(4, dtype=float)
    r44[0:3, 0:3] = cv.Rodrigues(rvec)[0]
    r44[0:3, 3] = tvec.flatten()
    return np.linalg.inv(r44)

def rvecTvecFromRotatingCamera(rvec, tvec, rotAxis, coordSys, rotAngleInDeg):
    """
    This function returns camera extrinsic parameters (rvec and tvec) given
    original rvec and tvec before rotating, the rotating axis, and the 
    rotating angle (using right-hand rule).

    Parameters
    ----------
    rvec : np.array, tuple, or list, must be three floats, or a 3x3 np.array.
        rvec of the camera before rotating
    tvec : np.array, tuple, or list, must be three floats.
    rotAxis : np.array, tuple, or list, must be three floats.
        rotating axis. If coordSys starts with 'g' or 'G' the rotAxis is 
        in global coordinate, otherwise in local coordinate.
    coordSys : string
        coordSys.strip()[0] == 'g' or 'G' means the rotAxis is in global 
        coordinate system otherwise it is in camera local coordinate system.
    rotAngleInDeg : float
        rotating angle in unit of degree, using right-hand rule

    Returns
    -------
    rvec : np.array (3 by 1)
        Rotational vector of the extrinsic parameters
    tvec : np.array (3 by 1)
        Translational vector of the extrinsic parameters
        
    Example
    -------
    See test_rvecTvecFromRotatingCamera()
    """
    # check inputs
    if type(rvec) == np.ndarray and rvec.shape == (3, 3):
        rvec = cv.Rodrigues(rvec.reshape(3,3))[0]
    rvec = np.array(rvec, dtype=float).reshape(3, 1)
    tvec = np.array(tvec, dtype=float).reshape(3, 1)
    # normalize rotAxis
    rotAxis = np.array(rotAxis, dtype=float).reshape(3, 1)
    rotAxis = rotAxis / np.linalg.norm(rotAxis)
    rotAxis = rotAxis * (rotAngleInDeg * np.pi / 180.)
    # calculate r44 and r44inv matrix from rvec and tvec
    r44 = np.eye(4, dtype=float)
    r44[0:3, 0:3] = cv.Rodrigues(rvec)[0]
    r44[0:3, 3] = tvec.flatten()
    r44inv = np.linalg.inv(r44)
    # calculate rotating axis in global axis
    if coordSys.strip()[0].lower() == 'g':
        rotAxis_g = np.array(rotAxis, dtype=float).reshape(3, 1)
    else:
        rotAxis_g = np.matmul(r44inv[0:3, 0:3], rotAxis.reshape(3, 1))
    # calculate the rotating matrix (3 by 3)
    rotMat = cv.Rodrigues(rotAxis_g)[0]
    # rotate --> r44inv
    r44inv[0:3, 0:3] = np.matmul(rotMat, r44inv[0:3, 0:3])
    # inverse back to r44
    r44 = np.linalg.inv(r44inv)
    # convert r44 to new_rvec and new_tvec
    new_rvec = cv.Rodrigues(r44[0:3, 0:3])[0].reshape(3, 1)
    new_tvec = r44[0:3, 3].reshape(3, 1)
    # return
    return new_rvec, new_tvec

    
def test_rvecTvecFromRotatingCamera():
    r44inv = np.eye(4, dtype=float)
    r44inv[0:3, 1] = np.array([0,0,-1.]) 
    r44inv[0:3, 2] = np.array([0,1,0.]) 
    r44inv[0:3, 3] = np.array([100, 200, 300.])
    r44 = np.linalg.inv(r44inv)
    rvec = cv.Rodrigues(r44[0:3, 0:3])[0].reshape(3, 1)
    tvec = r44[0:3, 3].reshape(3, 1)
    print("Initial state R44inv:")
    print(r44invOf(rvec, tvec))
    # rotate global z by 30 degrees
    rotAxis = [0, 0, 1.]
    coordSys = 'g'
    rotAngleInDeg = 30.0
    rvec, tvec = rvecTvecFromRotatingCamera(rvec, tvec, rotAxis, coordSys, rotAngleInDeg)    
    print("After rot global Z-30:")
    print(r44invOf(rvec, tvec))
    # rotate local x 10
    rotAxis = [1, 0, 0]
    coordSys = 'l'
    rotAngleInDeg = 10.0
    rvec, tvec = rvecTvecFromRotatingCamera(rvec, tvec, rotAxis, coordSys, rotAngleInDeg)    
    print("After rot local X-10:")
    print(r44invOf(rvec, tvec))
    # rotate local x 30
    rotAxis = [1, 0, 0]
    coordSys = 'l'
    rotAngleInDeg = 30.0
    rvec, tvec = rvecTvecFromRotatingCamera(rvec, tvec, rotAxis, coordSys, rotAngleInDeg)    
    print("After rot local X-30:")
    print(r44invOf(rvec, tvec))
    # rotate local x 50
    rotAxis = [1, 0, 0]
    coordSys = 'l'
    rotAngleInDeg = 50.0
    rvec, tvec = rvecTvecFromRotatingCamera(rvec, tvec, rotAxis, coordSys, rotAngleInDeg)    
    print("After rot local X-50:")
    print(r44invOf(rvec, tvec))
    # rotate global z -30
    rotAxis = [0, 0, 1.]
    coordSys = 'g'
    rotAngleInDeg = -30.0
    rvec, tvec = rvecTvecFromRotatingCamera(rvec, tvec, rotAxis, coordSys, rotAngleInDeg)    
    print("After rot global Z-(-30):")
    print(r44invOf(rvec, tvec))

    
    
        
        
        
        
        
        
        