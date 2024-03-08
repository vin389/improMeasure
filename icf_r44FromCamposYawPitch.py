import numpy as np
from inputs import input2
from npFromStr import npFromStr
from r44FromCamposYawPitch import r44FromCamposYawPitch

def icf_r44FromCamposYawPitch(
        campos: np.ndarray = None, 
        yaw: float = None,
        pitch: float = None,
        str_output_file: str = None):
    """
     This function runs an interactive procedure with the user to
     calculate the 4-by-4 matrix form of extrinsic parameters 
     of a camera according to camera yaw and pitch.
     Considering the world coordinate X-Y-Z where Z is upward,
     starting from an initial camera orientation (x,y,z) which 
     is (X,-Z,Y), that y is downward (-Z), rotates the camera y axis 
     (yaw, right-hand rule) then camera x axis (pitch, right-hand rule)
     in degrees.
       This function guarantees the camera axis x is always on world 
     plane XY (i.e., x has no Z components)
       You are supposed to provide the following information: 
     (1) Enter the camera position:
         (campos: 3x1 or 1x3 or (3,) numpy float array
         For example:
         1 1.732 1.5  or [1, 1.732, 1.5]  or c:/test/campos.csv
     (2) Enter the yaw:
         (in unit of degree, rotating about camera y axis.)
         (The initial camera x axis is world x axis.)
         (The initial camera y axis is world -z axis.)
         (The initial camera z axis is world y axis.)
         For example:
         30.0
     (3) Enter the pitch:
         (pitch angle in unit of degree, rotating about camera x axis.)
         For example:
         36.87
     (4) Enter the output file:
         (Full path of the output file.)
         For example:
         c:testcamera_1_r44.csv
         or cout for printing it on the screen.
     Example 1:
       1 1.732 1.5 
       30.0
       36.87
       c:testcamera_1_r44.csv
       You get the following data in the file:
       8.6602540378443871e-01,   -4.9999999999999989e-01,    6.7321191587137442e-17,   -2.5403784439091805e-05
       3.0000071456633115e-01,    5.1961647993585425e-01,   -7.9999892814850870e-01,    2.1934407532256931e-05
       3.9999946407425419e-01,    6.9281939477693022e-01,    6.0000142913266241e-01,   -2.4999647995268908e+00
       0.0000000000000000e+00,    0.0000000000000000e+00,    0.0000000000000000e+00,    1.0000000000000000e+00
    
    Parameters
    ----------
    campos : np.ndarray, optional
        camera position, in format of 3, 3x1, or 1x3 np.ndarray. 
        The default is None.
    yaw : float, optional
        yaw rotating about camera y axis, measured in degrees. 
         (The initial camera x axis is world x axis.)
         (The initial camera y axis is world -z axis.)
         (The initial camera z axis is world y axis.)
        The default is None.
    pitch : float, optional
        pitch angle rotating about camera x axis, 
        measured in degrees. The default is None.
    str_output_file : str, optional
        output file of the 4-by-4 transformation matrix
        can be cout if the output is the standard output
        
    Returns
    -------
    transformation matrix in a 4-by-4 np.ndarray (dtype=float) form 
        
    """
    if type(campos) == type(None):
        print("# ---------------------------------------------------")
        print("# (1) Enter the camera position:")
        print("#     (campos: 3x1 or 1x3 or (3,) numpy float array")
        print("#     For example:")
        print("#     1 1.732 1.5  or [1, 1.732, 1.5] or file c:/test/campos.csv (You need to type \"file\", a space, and the full path of the file name.)")
        str_campos = input2()
        campos = npFromStr(str_campos)
    if type(yaw) == type(None):
        print("# (2) Enter the yaw:")
        print("#     (in unit of degree, rotating about camera y axis.)")
        print("#     (The initial camera x axis is world x axis.)")
        print("#     (The initial camera y axis is world -z axis.)")
        print("#     (The initial camera z axis is world y axis.)")
        print("#     For example:")
        print("#     30.0")
        str_yaw = input2()
        yaw = float(str_yaw)
    if type(pitch) == type(None):
        print("# (3) Enter the pitch:")
        print("#     (pitch angle in unit of degree, rotating about camera x axis.)")
        print("#     For example:")
        print("#     36.87")
        str_pitch = input2()
        pitch = float(str_pitch)
    if type(str_output_file) == type(None):
        print("# (4) Output file:")
        print("#     (Full path of the output file.)")
        print("#     For example:")
        print("#     c:\\test\\camera_1_r44.csv")
        print("#     or cout for printing it on the screen.\n")
        str_output_file = input2()
    # ------
    r44 = r44FromCamposYawPitch(campos, yaw, pitch)
    # ------
    if (str_output_file == 'cout' or str_output_file == 'stdout'):
        r44_str = np.array2string(r44, separator=', ', precision=16)
        print(r44_str)
    else:
        footer='Extrinsic 4x4 matrix calculated from:' +\
               'camera position: %s, ' % str(campos.flatten()) +\
               'yaw: %25.16e, ' % yaw +\
               'pitch: %25.16e\n' % pitch
        np.savetxt(str_output_file, r44, fmt='%25.16e', delimiter=', ',
                   footer=footer)
    # return
    return r44
