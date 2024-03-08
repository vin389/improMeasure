from math import cos, sin, pi
import numpy as np

def r44FromCamposYawPitch(cameraPosition: np.ndarray, yaw: float, pitch: float):
    """
    Calculates the 4-by-4 matrix form of extrinsic parameters of a camera according to camera yaw and pitch.
    Considering the world coordinate X-Y-Z where Z is upward, 
    starting from an initial camera orientation (x,y,z) which is (X,-Z,Y), that y is downward (-Z), 
    rotates the camera y axis (yaw, right-hand rule) then camera x axis (pitch, right-hand rule) in degrees.
    This function guarantee the camera axis x is always on world plane XY (i.e., x has no Z components)
    Example:
        campos = np.array([ -100, -400, 10],dtype=float)
        yaw = 15.945395900922847; pitch = 13.887799644071938;
        r44Cam = r44FromCamposYawPitch(campos, yaw, pitch)
        # r44Cam would be 
        # np.array([[ 0.961, -0.275,  0.000, -1.374],
        #           [ 0.066,  0.231, -0.971,  108.6],
        #           [ 0.267,  0.933,  0.240,  397.6],
        #           [ 0.000,  0.000,  0.000,  1.000]])
        
    Parameters
    ----------
    cameraPosition: TYPE np.array((3,3),dtype=float)
        camera position in the world coordinate 
    yaw: TYPE float
        camera yaw along y axis (right-hand rule) (in degree), clockwise is positive
        E.g., camera aiming +Y-axis is yaw of 0 here; aiming +X-axis is yaw of 90 here.
    pitch: TYPE float
        camera pitch along x axis (right-hand rule) (in degree), upward is positive

    Returns
    -------
    TYPE: np.array((4,4),dtype=float)
        the 4-by-4 matrix form of the extrinsic parameters
    """
    # camera vector (extrinsic)
    vxx = cos(yaw * pi / 180.)
    vxy = -sin(yaw * pi / 180.)
    vxz = 0
    vx_cam = np.array([vxx, vxy, vxz], dtype=np.float64)
    vzx = sin(yaw * pi / 180.) * cos(pitch * pi / 180.)
    vzy = cos(yaw * pi / 180) * cos(pitch * pi / 180.)
    vzz = sin(pitch * pi / 180.)
    vz_cam = np.array([vzx, vzy, vzz], dtype=np.float64)
    vy_cam = np.cross(vz_cam, vx_cam)
    r44inv = np.eye(4, dtype=np.float64)
    r44inv[0:3, 0] = vx_cam[0:3]
    r44inv[0:3, 1] = vy_cam[0:3]
    r44inv[0:3, 2] = vz_cam[0:3]
    r44inv[0:3, 3] = cameraPosition[0:3]
    r44 = np.linalg.inv(r44inv)
    return r44
