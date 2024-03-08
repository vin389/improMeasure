import numpy as np
import cv2 as cv

def r44FromCamposAndAim(cameraPosition, aimPoint):
    """
    Calculates the 4-by-4 matrix form of extrinsic parameters of a camera according to camera position and a point it aims at.
    Considering the world coordinate X-Y-Z where Z is upward, 
    starting from an initial camera orientation (x,y,z) which is (X,-Z,Y), that y is downward (-Z), 
    rotates the camera so that it aims a specified point (aim)
    This function guarantee the camera axis x is always on world plane XY (i.e., x has no Z components)
    Example:
        campos = np.array([ -100, -400, 10],dtype=float)
        aim = np.array([0, -50, 100],dtype=float)
        r44Cam = r44FromCamposAndAim(campos,aim)
        # r44Cam would be 
        # np.array([[ 0.961, -0.275,  0.000, -1.374],
        #           [ 0.066,  0.231, -0.971,  108.6],
        #           [ 0.267,  0.933,  0.240,  397.6],
        #           [ 0.000,  0.000,  0.000,  1.000]])
        
    Parameters
    ----------
    cameraPosition: TYPE np.array((3,3),dtype=float)
        camera position in the world coordinate 
    aimPoint: TYPE np.array((3,3),dtype=float)
        the aim that the camera is aiming at

    Returns
    -------
    TYPE: np.array((4,4),dtype=float)
        the 4-by-4 matrix form of the extrinsic parameters
    """
    # camera vector (extrinsic)
    vz_cam = aimPoint - cameraPosition
    vy_cam = np.array([0,0,-1], dtype=np.float64)
    vz_cam = vz_cam / np.linalg.norm(vz_cam)
    vx_cam = np.cross(vy_cam, vz_cam)
    vy_cam = np.cross(vz_cam, vx_cam)
    vx_cam = vx_cam / np.linalg.norm(vx_cam)
    vy_cam = vy_cam / np.linalg.norm(vy_cam)
    vz_cam = vz_cam / np.linalg.norm(vz_cam)
    r44inv = np.eye(4, dtype=np.float64)
    r44inv[0:3, 0] = vx_cam[0:3]
    r44inv[0:3, 1] = vy_cam[0:3]
    r44inv[0:3, 2] = vz_cam[0:3]
    r44inv[0:3, 3] = cameraPosition[0:3]
    r44 = np.linalg.inv(r44inv)
    return r44
