import os, re
from math import atan2, pi, sin, cos #, tan, atan
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


def r44FromCamposYawPitch(cameraPosition, yaw, pitch):
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


def cameraParametersToString(imgSize, rvec, tvec, cmat, dvec):
    # generate a string that is easy to read
    imgSize = np.array(imgSize).flatten()
    r33, jmat= cv.Rodrigues(rvec.reshape(-1))
    r44 = np.eye(4, dtype=float)
    r44[0:3,0:3] = r33
    r44[0:3,3] = tvec.reshape(-1)
    r44inv = np.linalg.inv(r44)
    theStr = ''
    theStr += '# Image size (width, height) is (%d, %d)\n' % (imgSize[0], imgSize[1])
    theStr += '# rvec_x = %24.16e\n' % (rvec[0])
    theStr += '# rvec_y = %24.16e\n' % (rvec[1])
    theStr += '# rvec_z = %24.16e\n' % (rvec[2])
    theStr += '# tvec_x = %24.16e\n' % (tvec[0])
    theStr += '# tvec_y = %24.16e\n' % (tvec[1])
    theStr += '# tvec_z = %24.16e\n' % (tvec[2])
    theStr += '# r44 matrix:\n'
    for i in range(4):
        theStr += '#   %24.16e %24.16e %24.16e %24.16e\n' % \
                  (r44[i,0], r44[i,1], r44[i,2], r44[i,3])
    theStr += '# inverse of r44 matrix:\n'
    for i in range(4):
        theStr += '#   %24.16e %24.16e %24.16e %24.16e\n' % \
                  (r44inv[i,0], r44inv[i,1], r44inv[i,2], r44inv[i,3])
    theStr += "# fx = %24.16e (pixels)\n" % (cmat[0,0])
    theStr += '# fy = %24.16e (pixels)\n' % (cmat[1,1])
    theStr += '# cx = %24.16e (pixels)\n' % (cmat[0,2])
    theStr += '# cy = %24.16e (pixels)\n' % (cmat[1,2])
    distStr = ['k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'k5', 'k6', 
               's1', 's2', 's3', 's4', 'taux', 'tauy']
    for i in range(dvec.reshape(-1).size):
        if i <= 3 or dvec.reshape(-1)[i] != 0.0:
            theStr += '# %s = %24.16e\n' % (distStr[i], dvec.reshape(-1)[i])
    return theStr


def readCamera(filename=None, example=None):
    """
    Reads camera parameters (rvec, tvec, cmat, dvec) from file. Data should be
    in 1-column vector in text format composed with rvec (3 elements), tvec (3
    elements), cmat (9 (3 by 3) elements), and dvec (4, 5, 8, 12, or 14 
    elements) all combined in a single column vector.  

    Parameters
    ----------
    filename : str
        file name that contains the rvec, tvec, cmat, and dvec. 
        If not given, this function asks the user.

    Returns
    -------
    imgSize : np.ndarray (shape: 2)
    rvec : np.ndarray (shape: 3)
        rvec rotational vector of the extrinsic parameters
    tvec : np.ndarray (shape: 3)
        tvec translational vector of the extrinsic parameters
    cmat : np.ndarray (shape: (3,3))
        cmat the camera matrix 
    dvec : np.ndarray (shape: n, n can be 4, 5, 8, 12, or 14)
        dvec the distortion coefficients (k1, k2, p1, p2[, k3, k4, k5, k6[
            , s1, s2, s3, s4[, taux, tauy]]]) 
    Example:
    --------
    Here is an example of the file of a camera:
    # Start of the example of the file
    # An example of camera parameters
    3840
    2160
    1.123
    -0.135
    0.135
    -700.
    800.
    5400.
    8000.
    0.
    2999.5
    0.
    8000.
    1999.5
    0.
    0.
    1.
    -0.02
    0.
    0.
    0.
    # End of the example of the file
    imgSize, rvec, tvec, cmat, dvec = readCamera('camera_1.txt')
    

    """
    imgSize = np.array([]) # mat[0:2]
    rvec = np.array([])   # mat[2:5]
    tvec = np.array([])   # mat[5:8]
    cmat = np.array([])   # mat[8:17]
    dvec = np.array([])   # mat[17:]
    # if filename is given
    if type(filename) == str:
        try:
            mat = np.loadtxt(filename, delimiter=',')
            mat = mat.flatten()
            if mat.size < 21:
                print("# Error. readCamera(): Matrix size is too small for camera parameters.")
                print("#  Matrix size should be >= 21")
                print("#  First 2 [0:2] for image size (width, height), ")
                print("#  followed by [2:8] for rvec and tvec,  ")
                print("#  3x3 (9) [8:17] for camera matrix, and at least 4 for")
                print("#  distortion coefficients [17:21 or more] (k1,k2,p1,p2[,k3[")
                print("#                                ,k4,k5,k6[")
                print("#                                ,s1,s2,s3,s4[")
                print("#                                ,taux,tauy]]]]")
            else:
                imgSize = mat[0:2].copy()
                rvec = mat[2:5].copy()
                tvec = mat[5:8].copy()
                cmat = mat[8:17].copy().reshape(3,3)
                dvec = mat[17:].copy()
                return imgSize.astype(int).flatten(), rvec.reshape(3, 1), tvec.reshape(3, 1), cmat.reshape(3, 3), dvec.reshape(1, -1)
        except Exception as e:
            print("# Error: readCamera() cannot read data from file (%s)."
                   % (filename))
            print("# Exception is", e)
            print("# The file needs to be in text format.")
    # ask user if filename is empty
    while (imgSize.size != 2 or rvec.size != 3 or tvec.size != 3 \
           or cmat.size != 9 or dvec.size < 4):
        print("\n# How do you want to input camera parameters:")
        print("#     file: reading matrix from file (text format)")
        print("#       which matrix size should be >= 19")
        print("#  First 2 [0:2] for image size (width, height), ")
        print("#  followed by [2:8] for rvec and tvec,  ")
        print("#  3x3 (9) [8:17] for camera matrix, and at least 4 for")
        print("#  distortion coefficients [17:21 or more] (k1,k2,p1,p2[,k3[")
        print("#                                ,k4,k5,k6[")
        print("#                                ,s1,s2,s3,s4[")
        print("#                                ,taux,tauy]]]]")
        print("#       For example: ")
        print("#           file")
        if (type(example) == type(None)):
            print("#           ..\\..\\examples\\camera_1.txt")
        else:
            print("#           %s" % (example))
        print("#    manual")
        print("#       for example:")
        print("#         manual")
        print("#           # imgSize")
        print("#           3840, 2160 ")
        print("#           # rvec")
        print("#           0, 0, 0 ")
        print("#           # tvec")
        print("#           0, 0, 0 ")
        print("#           # camera matrix")
        print("#           1800, 0, 959.5")
        print("#           0, 1800, 539.5")
        print("#           0,0,1")
        print("#           # distortion coefficients")
        print("#           0.0321, 0, 0, 0")
        uInput = input2().strip()
        if (uInput == 'file'):
            try:
                print("#  Enter file of camera parameters:")
                filename = input2()
                mat = np.loadtxt(filename, delimiter=',')
                mat = mat.flatten()
                if mat.size < 21:
                    print("# Error. readCamera(): Matrix size is too small for camera parameters.")
                    print("#  Matrix size should be >= 19")
                    print("#  First 2 [0:2] for image size (width, height), ")
                    print("#  followed by [2:8] for rvec and tvec,  ")
                    print("#  3x3 (9) [8:17] for camera matrix, and at least 4 for")
                    print("#  distortion coefficients [17:21 or more] (k1,k2,p1,p2[,k3[")
                    print("#                                ,k4,k5,k6[")
                    print("#                                ,s1,s2,s3,s4[")
                    print("#                                ,taux,tauy]]]]")
                else:
                    imgSize = mat[0:2].copy()
                    rvec = mat[2:5].copy()
                    tvec = mat[5:8].copy()
                    cmat = mat[8:17].copy().reshape(3,3)
                    dvec = mat[17:].copy()
            except Exception as e:
                print("# Error: readCamera() cannot read matrix from file (%s)."
                       % (filename))
                print("# Exception is", e)
                print("# Try again.")
                continue
        if (uInput[0:6] == 'manual'):
            print("#  Enter 2 integers for image size (width and height):")
            datInput = input2("").strip()
            datInput = re.split(',| ', datInput)
            datInput = list(filter(None, datInput))
            imgSize = np.zeros(2, dtype=float)
            print("#  Enter 3 reals for rvec:")
            datInput = input2("").strip()
            datInput = re.split(',| ', datInput)
            datInput = list(filter(None, datInput))
            rvec = np.zeros(3, dtype=float)
            for i in range(3):
                rvec[i] = float(datInput[i])
            print("#  Enter 3 reals for tvec:")
            datInput = input2("").strip()
            datInput = re.split(',| ', datInput)
            datInput = list(filter(None, datInput))
            tvec = np.zeros(3, dtype=float)
            for i in range(3):
                tvec[i] = float(datInput[i])
            print("#  Enter 3x3 reals for camera matrix:")
            print("#    three elements (one row) in a line:")
            print("#    For example:")
            print("#           1800, 0, 959.5")
            print("#           0, 1800, 539.5")
            print("#           0,0,1")
            cmat = np.eye(3, dtype=float)
            for i in range(3):
                datInput = input2("").strip()
                datInput = re.split(',| ', datInput)
                datInput = list(filter(None, datInput))
                for j in range(3):
                    cmat[i,j] = datInput[j]
            print("#  Enter n reals for distortion coefficients (n can be 4, 5, 8, 12, or 14):")
            print("#  distortion coefficients: k1,k2,p1,p2[,k3[,k4,k5,k6[,s1,s2,s3,s4[,taux,tauy]]]]")
            datInput = input2("").strip()
            datInput = re.split(',| ', datInput)
            datInput = list(filter(None, datInput))
            nDistCoeffs = len(datInput)
            dvec = np.zeros(nDistCoeffs, dtype=float)
            for i in range(nDistCoeffs):
                dvec[i] = float(datInput[i])
        if (imgSize.size == 2 and rvec.size == 3 and tvec.size == 3 
            and cmat.size == 9 and dvec.size >= 4):
            theStr = cameraParametersToString(imgSize, rvec, tvec, cmat, dvec)
            print(theStr)
            break
        else:
            print("# readCamera(): Invalid size(s) of parameters.")
            print(imgSize.astype(int))
            print(rvec.reshape(3, 1))
            print(tvec.reshape(3, 1))
            print(cmat.reshape(3, 3))
            print(dvec.reshape(1, -1))
    return imgSize.astype(int).flatten(), rvec.reshape(3, 1), tvec.reshape(3, 1), cmat.reshape(3, 3), dvec.reshape(1, -1)
        

def writeCamera(savefile, imgSize, rvec, tvec, cmat, dvec):
    imgSize = np.array(imgSize).flatten()
    rvec = np.array(rvec).reshape(-1, 1)
    tvec = np.array(tvec).reshape(-1, 1)
    cmat = np.array(cmat).reshape(3, 3)
    dvec = np.array(dvec).reshape(1, -1)
    if imgSize.size != 2 or rvec.size != 3 or tvec.size != 3 \
        or cmat.size != 9 or dvec.size < 4:
        print("# Error. writeCamera(): Invalid sizes of parameters.")
        print("# imgSize size should be 2 but is %d" % np.array(imgSize).size)
        print("# rvec size should be 3 but is %d" % (rvec.size))
        print("# tvec size should be 3 but is %d" % (tvec.size))
        print("# cmat size should be 9 but is %d" % (cmat.size))
        print("# dvec size should be >=4 but is %d" % (dvec.size))
        return
    # generate a string that is easy to read
    # if savefile is cout or stdout, print to screen
    if savefile == 'cout' or savefile == 'stdout':
        _savefile = '__temp_writeCamera__.txt'
    else: 
        _savefile = savefile
    camAll = np.zeros(2+3+3+9+dvec.size, dtype=float)
    camAll[0:2] = np.array(imgSize).flatten()
    camAll[2:5] = rvec.flatten();
    camAll[5:8] = tvec.flatten();
    camAll[8:17] = cmat.flatten();
    camAll[17:] = dvec;
    theStr = cameraParametersToString(imgSize, rvec, tvec, cmat, dvec)
    np.savetxt(_savefile, camAll, fmt='%24.16e', delimiter=' , ',
            header='Camera paramters. imgSize, rvec, tvec, cmat (flatten), dvec', 
            footer=theStr)
    if savefile == 'cout' or savefile == 'stdout':
        with open(_savefile, 'r') as file:
            content = file.read()
            print(content)
#            os.remove(_savefile)   
    return         


def input2(prompt=""):
    """
    This function is similar to Python function input() but if the returned
    string starts with a hashtag (#) this function ignores the line of the
    strin and runs the input() function again.
    The head spaces and tail spaces are removed as well.
    This function only allows user to edit a script for a series of input,
    but also allows user to put comments by starting the comments with a
    hashtag, so that the input script is earier to understand.
    For example, a BMI converter could run in this way:
    /* -------------------------------
    1.75  (user's input)
    70    (user's input)
    The BMI is 22.9
    --------------------------------- */
    The user can edit a file for future input:
    /* ---------------------------------
    # This is an input script for a program that calculates BMI
    # Enter height in unit of meter
    1.75
    # Enter weight in unit of kg
    70

    Parameters
        prompt  A String, representing a default message before the input.
    --------------------------------- */
    """
    theInput = ""
    if len(prompt) == 0:
        thePrompt = ""
    else:
        thePrompt = "# " + prompt
        print(thePrompt)
    # run the while loop of reading
    while(True):
        theInput = input()
        theInput = theInput.strip()
        if (len(theInput) == 0):
            continue
        if (theInput[0] == '#'):
            continue
        break
    # remove everything after the first #
    if theInput.find('#') >= 0:
        theInput = theInput[0:theInput.find('#')]
    return theInput

def inputs(prompt=''):
    """
    This function (inputs) is similar to input() but it allows multiple
    lines of input. 
    The key [Enter] does not ends the input. This function reads 
    inputs line by line until a Ctrl-D (or maybe Ctrl-Z), 'end', or 
    'eof' is entered. 
    This function returns a list of strings. 
    
    Parameters
    ----------
    prompt : TYPE, optional
        DESCRIPTION. The default is ''.

    Returns
    -------
    contents : List of strings
        A list of strings. Each string is a line of input.
    """
    print(prompt, end='')
    contents = []
    while True:
        try:
            line = input()
            if line.strip().lower() == 'end' or line.strip().lower() == 'eof':
                raise Exception
        except:
            break
        contents.append(line)
    return contents


def plot_triangle_on_axes(p1, p2, p3, color, alpha=1.0, ax=[]):
    """
    This function plots a triangle in a 3D axes. 
    If the axes (ax) is not given, this function creates a new one and returns
    it after plotting.
    Example:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection = '3d')
        p1 = (0, 0, 0)
        p2 = (1, 0, 0)
        p3 = (0.5, 1, 0)
        plot_triangle_on_axes(p1, p2, p3, color='blue', ax=ax)
        p4 = (0.5, 0.5, 1)
        plot_triangle_on_axes(p4, p2, p3, color='red', ax=ax)
        plt.show()
    """
    
    # import package
    import matplotlib.pyplot as plt
    #
    if ax == []:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    xs = [p1[0], p2[0], p3[0]]
    ys = [p1[1], p2[1], p3[1]]
    zs = [p1[2], p2[2], p3[2]]
    ax.plot_trisurf(xs, ys, zs, color=color, alpha=alpha)
    return ax

def plot_quad_on_axes(p1, p2, p3, p4, color, alpha=1.0, ax=[]):
    """
    This function plots a quadrilateral in a 3D axes. 
    If the axes (ax) is not given, this function creates a new one and returns
    it after plotting.
    Example:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection = '3d')
        p1 = (0, 1, 0)
        p2 = (1, 1, 0)
        p3 = (0.9, 2, 0)
        p4 = (0.1, 2, 0)
        plot_quad_on_axes(p1, p2, p3, p4, color='blue', ax=ax)
        plt.show()
    """
    # import package
    import matplotlib.pyplot as plt
    #
    if ax == []:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    xs = np.array([[p1[0], p2[0]], [p4[0], p3[0]]])
    ys = np.array([[p1[1], p2[1]], [p4[1], p3[1]]])
    zs = np.array([[p1[2], p2[2]], [p4[2], p3[2]]])
    ax.plot_surface(xs, ys, zs, color=color, alpha=alpha)
    return ax


def npFromStrings(theStrList: list):
    """
    Converts a list of strings (which only contains floats) to a numpy 
    float array (in 1D). The separator can be ',', ' ', '\t', '\n', 
    '(', ')', '[', ']', '{', '}'. The 'nan' or 'na' would be considered 
    as np.nan. 
    The returned numpy will be in 1D. 
    For example:
        npFromStrings(['1.2 , 2.3', 
                       'nan \n 4.5'])
            returns array([1.2, 2.3, nan, 4.5])
    """
    theType = type(theStrList)
    if theType == str:
        return npFromString(theStrList.strip())
#   elif theType == list or theType == tuple:
    else: 
        # assuming the type is list or tuple, or 
        # other types that runs with the following for loop
        _str = ''
        for i in theStrList:
            _str += (str(i).strip() + '\n')
        return npFromString(_str)


def npFromString(theStr: str):
    """
    Converts a string (which only contains floats) to a numpy 
    float array (in 1D). The separator can be ',', ' ', '\t', '\n', 
    '(', ')', '[', ']', '{', '}'. The 'nan' or 'na' would be considered 
    as np.nan. 
    The returned numpy will be in 1D. 
    For example:
        npFromString('1.2 , 2.3 \t nan \n 4.5')
            returns array([1.2, 2.3, nan, 4.5])
    """
    if type(theStr) == str:
        _str = theStr.strip()
    elif type(theStr) == list or type(theStr) == tuple:
        return npFromStrings(theStr)
    else: 
        _str = str(theStr)
    _str = _str.replace(',', ' ').replace(';', ' ').replace('[', ' ')
    _str = _str.replace(']', ' ').replace('na ', 'nan').replace('\n',' ')
    _str = _str.replace('(', ' ').replace(')', ' ')
    _str = _str.replace('{', ' ').replace('}', ' ')
    _str = _str.replace('n/a', 'nan').replace('#N/A', 'nan')
    _str = _str.replace('np.nan', 'nan').replace('numpy.nan', 'nan')
    theMat = np.fromstring(_str, sep=' ')
    return theMat   


def uigetfile(fileDialogTitle='Select the file to open', initialDirectory='/', fileTypes = (('All files', '*.*'), ('TXT files', '*.txt;*.TXT'), ('JPG files', '*.jpg;*.JPG;*.JPEG;*.jpeg'), ('BMP files', '*.bmp;*.BMP'), ('Csv files', '*.csv'), ('opencv-supported images', '*.bmp;*.BMP;*.pbm;*.PBM;*.pgm;*.PGM;*.ppm;*.PPM;*.sr;*.SR;*.ras;*.RAS;*.jpeg;*.JPEG;*.jpg;*.JPG;*.jpe;*.JPE;*.jp2;*.JP2;*.tif;*.TIF;*.tiff;*.TIFF'), )):
    #
    import tkinter as tk
    from tkinter import filedialog
    #
    filePath = []
    fileName = []    
    tmpwin = tk.Tk()
    tmpwin.lift()    
    #window.iconify()  # minimize to icon
    #window.withdraw()  # hide it 
    fullname = filedialog.askopenfilename(title=fileDialogTitle, initialdir=initialDirectory, filetypes=fileTypes)        
    tmpwin.destroy()
    if fullname:
        allIndices = [i for i, val in enumerate(fullname) if val == '/']
        filePath = fullname[0 : 1+max(allIndices)]
        fileName = fullname[1+max(allIndices) : ]
    return filePath, fileName


class Camera:
    """! The Camera class.
    
    Defines a camera object. 
    Its data include:
        self.imgSize which is a tuple (width, height)
        self.rvec which is a 3-by-1 numpy matrix in float64.
        self.tvec which is a 3-by-1 numpy matrix in float64.
        self.cmat which is a 3-by-3 numpy matrix in float64.
        self.dvec which is a 1-by-n (4 <= n <= 14) in float64.
    Its functions include:
        __init__(rvec=None, tvec=None, cmat=None, dvec=None, imgSize=None):
        initByKeyin(keyin: str = '')
        initByNumpyFile(filepath='')
        initByAsk()
        fovx(), fovy(), cx_ratio(), cy_ratio(), 
        fx(), fy(), cx(), cy(), k1(), k2(), p1(), p2(), k3(), k4(),
        k5(), k6(), 
        rmat(), rmat44(), campos(), 
        setRvecTvecByPosAim(pos, aim), 
        rotateCamera(rotAxis, coordSys, rotAngleInDeg),
        saveToFile(file), 
        loadFromFile(file),
        plotCameraPyramid(axes=[]),
        undistort(img)
        __str__()
        
    """
    def __init__(self, imgSize=None, rvec=None, tvec=None, cmat=None, dvec=None):
        # image size
        if imgSize is None:
#            self.imgSize = (self.cmat[0, 2] * 2 + 1, self.cmat[1, 2] * 2 + 1)
            self.imgSize = (-1, -1)
        else:
            self.imgSize = tuple(imgSize)
        # rvec
        if rvec is None:
            self.rvec = np.zeros((3, 1))
        else:
            rvec = np.array(rvec, dtype=float)
            if rvec.shape[0] == 3 and rvec.shape[1] == 3:
                self.rvec = cv.Rodrigues(rvec)[0].reshape((3, 1))
            else:
                self.rvec = rvec.astype(float)
        # tvec
        if tvec is None:
            self.tvec = np.zeros((3, 1), dtype=float)
        else:
            self.tvec = np.array(tvec, dtype=float).reshape((3, 1))
        # cmat
        if cmat is None:
#            self.cmat = np.array([[1920,0,1919.5],[0,1920,1079.5],[0,0,1]], dtype=float)
            self.cmat = np.eye(3, dtype=float)
        else:
            self.cmat = np.array(cmat, dtype=float).reshape((3, 3))
        # dvec
        if dvec is None:
            self.dvec = np.zeros((5,1), dtype=float)
        else:
            self.dvec = np.array(dvec, dtype=float).reshape((-1, 1))

    def initByKeyin(self, keyin: str = ''):
        """
        Allows user to input image size (width and height) (2 integers), rvec (3 reals), tvec (3 reals), cmat (9 reals), 
        and dvec (4 to 14 reals), (totally 21 to 31 reals) to initialize a 
        Camera object. The key input ends by Ctrl-D or Ctrl-Z. 
        You can also directly give key-in string as an argument.
        
        Returns
        -------
        None.
        """
        if len(keyin) <= 1:
            print("# Enter 21 to 31 reals to initialize a camera. These reals are image size (width and height) (2), rvec (3), tvec (3), cmat(9), dvec(4 to 14) and Ctrl-D (or maybe Ctrl-Z):")
            keyin = inputs()
        vec = npFromString(keyin)
        vec = vec.flatten()
        # vec length (extrinsic + intrinsic) should be between 21 and 31
        # [2 + 3 + 3 + 9 + 4, 2 + 3 + 3 + 9 + 14]
        if vec.size > 31:
            vec = vec[0:31]
        if vec.size < 21:
            _vec = np.zeros(21, dtype=float)
            _vec[0:vec.size] = vec[:]
            vec = _vec
        self.imgSize = vec[0:2].astype(int).flatten()
        self.rvec = vec[2:5].reshape((3, 1))
        self.tvec = vec[5:8].reshape((3, 1))
        self.cmat = vec[8:17].reshape((3, 3))
        self.dvec = vec[17:].reshape((1, -1))

    def initByNumpyFile(self, filepath=''):
        """
        Initializes the Camera by reading a Numpy file that contains a series of reals. The ordering is
        imgSize (2 reals in format but are numerically integers), rvec (3 reals), tvec (3 reals), cmat (9 reals), and dvec (4 to 14 reals).
        If the file contains more than 31 reals (no matter what dimensional shape it is), 
        the first 31 reals are used. But if the shape is not 1-D, or is 1-D but is longer than 31 reals, 
        this functions displays a warning message.
        """
        if len(filepath) <= 1 or os.path.exists(filepath) == False:
            # ask user to assign file path
            print("# Enter the camera file (.npy or .csv) (Enter . to switch to graphical file dialog): ")
            filepath = input()
        if len(filepath) <= 1 or os.path.exists(filepath) == False:
            # ask user to assign file path
            uigf = uigetfile()
            filepath = os.path.join(uigf[0], uigf[1])
        if len(filepath) <= 1 or os.path.exists(filepath) == False:
            print("# Invalid input. The camera object is not set.")
            return
        print("# Reading parameters from file %s ..." % filepath)
        imgSize, rvec, tvec, cmat, dvec = readCamera(filepath)
        self.imgSize = imgSize.astype(int).flatten()
        self.rvec = rvec.reshape((3, 1))
        self.tvec = tvec.reshape((3, 1))
        self.cmat = cmat.reshape((3, 3))
        self.dvec = dvec.reshape((1, -1))
        print("# Set camera to ", str(self))
        
    def initByAsk(self):
        print("# How do you want to define the camera: ")
        print("#  1. By key-in (or copy-and-paste) values,")
        print("#  2. By reading from a text file (through np.loadtxt): ")
        opt = input()
        if int(opt) == 1:
            self.initByKeyin()
        elif int(opt) == 2:
            self.initByNumpyFile()
        else:
            print("# Invalid selection. Request ignored.")
        return
        
    def fovx(self): 
        dy = self.cmat[0, 0]
        dx = self.imgSize[0] / 2
        return atan2(dx, dy) * 180. / pi
    def fovy(self):
        dy = self.cmat[1, 1]
        dx = self.imgSize[1] / 2
        return atan2(dx, dy) * 180. / pi
    def cx_ratio(self):
        # cx_ratio 0.5 indicates the principal point is at the horizonal center 
        cx = self.cmat[0, 2]
        return (cx + 0.5) / self.imgSize[0]
    def cy_ratio(self):
        # cy_ratio 0.5 indicates the principal point is at the vertical center 
        cy = self.cmat[1, 2]
        return (cy + 0.5) / self.imgSize[1]
    def fx(self):
        return self.cmat[0, 0]
    def fy(self):
        return self.cmat[1, 1]
    def cx(self):
        return self.cmat[0, 2]
    def cy(self):
        return self.cmat[1, 2]
    def k1(self):
        return self.dvec[0, 0]
    def k2(self):
        return self.dvec[0, 1]
    def p1(self):
        return self.dvec[0, 2]
    def p2(self):
        return self.dvec[0, 3]
    def k3(self):
        return self.dvec[0, 4]
    def k4(self):
        return self.dvec[0, 5]  
    def k5(self):
        return self.dvec[0, 6]    
    def k6(self):
        return self.dvec[0, 7]

    def rmat(self):
        return cv.Rodrigues(self.rvec)[0]
    
    def rmat44(self):
        r44 = np.eye(4, dtype=float)
        r44[0:3, 0:3] = cv.Rodrigues(self.rvec)[0]
        r44[0:3, 3] = self.tvec[0:3,0].copy()
        return r44
    
    def campos(self):
        # return the camera position
        r44inv = np.linalg.inv(self.rmat44())
        return r44inv[0:3, 3].reshape((3, 1))
    
    def setRvecTvecByPosAim(self, pos, aim):
        rvec, tvec = rvecTvecFromPosAim(pos, aim)
        self.rvec = rvec
        self.tvec = tvec
        return
    
    def setRvecTvecByPosYawPitch(self, pos, yaw, pitch):
        r44 = r44FromCamposYawPitch(pos, yaw, pitch)
        rvec, tvec = rvecTvecFromR44(r44)
        self.rvec = rvec
        self.tvec = tvec
        return
    
    def rotateCamera(self, rotAxis, coordSys, rotAngleInDeg):
        """
        This function returns camera extrinsic parameters (rvec and tvec) 
        which rotate along the given axis about the current camera position. 
        The camera does not move, only rotation, given the rotating axis, 
        and the rotating angle (using right-hand rule).
        This function calls rvecTvecFromRotatingCamera() in 
        rvecTvecFromRotatingCamera.py

        Parameters
        ----------
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
        None.

        """
        new_rvec, new_tvec = rvecTvecFromRotatingCamera(
            self.rvec, self.tvec, rotAxis, coordSys, rotAngleInDeg)
        self.rvec = new_rvec
        self.tvec = new_tvec
        return
    
    def plotCameraPyramid(self, color='green', alpha=0.5, axes=[]):
        # import packages
        import matplotlib.pyplot as plt
        #
        pyramid_height = 100.0 # physical size, e.g., mm or m
        r44inv = np.linalg.inv(self.rmat44())
        vx = np.array(r44inv[0:3,0]).flatten()
        vy = np.array(r44inv[0:3,1]).flatten()
        vz = np.array(r44inv[0:3,2]).flatten()
        cp = np.array(r44inv[0:3,3]).flatten()
        fac = pyramid_height / self.fx()
        h = self.imgSize[1]
        w = self.imgSize[0]
        p1 = cp + fac * (vx * (-self.cx()) + 
                         vy * (-self.cy()) + 
                         vz * self.fx())
        p2 = p1 + fac * vy * h
        p3 = p2 + fac * vx * w
        p4 = p3 + fac * vy * (-h)
        axes = plot_triangle_on_axes(p1, p2, cp, color=color, alpha=alpha, ax=axes)
        axes = plot_triangle_on_axes(p2, p3, cp, color=color, alpha=alpha, ax=axes)
        axes = plot_triangle_on_axes(p3, p4, cp, color=color, alpha=alpha, ax=axes)
        axes = plot_triangle_on_axes(p4, p1, cp, color=color, alpha=alpha, ax=axes)
        # set axes limits (n times larger than it is required)
        n = 5
        xmin = min(p1[0], p2[0], p3[0], p4[0], cp[0])
        xmax = max(p1[0], p2[0], p3[0], p4[0], cp[0])
#        ymin = min(p1[1], p2[1], p3[1], p4[1], cp[1])
#        ymax = max(p1[1], p2[1], p3[1], p4[1], cp[1])
#        zmin = min(p1[2], p2[2], p3[2], p4[2], cp[2])
#        zmax = max(p1[2], p2[2], p3[2], p4[2], cp[2])
        xspan = n * max(cp[0] - xmin, xmax - cp[0])
#        yspan = n * max(cp[1] - ymin, ymax - cp[1])
#        zspan = n * max(cp[2] - zmin, zmax - cp[2])
        axes.set_xlim([cp[0] - xspan, cp[0] + xspan])
        axes.set_ylim([cp[1] - xspan, cp[1] + xspan])
        axes.set_zlim([cp[2] - xspan, cp[2] + xspan])
        axes.set_box_aspect([1, 1, 1])
        axes.set_xlabel('Global X')
        axes.set_ylabel('Global Y')
        axes.set_zlabel('Global Z')
        #
        plt.show()
        return axes

    def saveToFile(self, file):
        writeCamera(file, self.imgSize, self.rvec, self.tvec, self.cmat, self.dvec)
        return
    
    def loadFromFile(self, file):
        imgSize, rvec, tvec, cmat, dvec = readCamera(file)
        self.imgSize = imgSize.astype(int).flatten()
        self.rvec = rvec.reshape((3, 1))
        self.tvec = tvec.reshape((3, 1))
        self.cmat = cmat.reshape((3, 3))
        self.dvec = dvec.reshape((1, -1))
        return
    
    def undistort(self, img):
        imgud = cv.undistort(img, self.cmat, self.dvec)
        return imgud

    def __str__(self):
        # s = cameraParametersToString(self.imgSize, self.rvec, 
        #     self.tvec, self.cmat, self.dvec) 
        # s = ''
        # s += "# Camera information:\n"
        # s += ("#  Image size: " + str(self.imgSize) + "\n")
        # s += "#   FOV(X) is %f degrees.\n" % (self.fovx())
        # s += "#   FOV(Y) is %f degrees.\n" % (self.fovy())
        # s += "# Camera matrix:\n"
        # s += (str(self.cmat) + "\n")
        # s += "# Distortion coefficient: "
        # s += str(self.dvec.flatten())
        s = cameraParametersToString(self.imgSize, self.rvec, 
            self.tvec, self.cmat, self.dvec)
        return s

def test_Camera():
    imgsstr = "6000 4000"
    cmatstr = "6.8149575235495613e+03 0.0000000000000000e+00 3.2687868735327825e+03 0.0000000000000000e+00 7.0203284901151501e+03 2.5479944244095450e+03 0 0 1"
    dvecstr = "3.6098413064799256e-01 -2.2185764954226981e+00 1.6928010899348759e-02 -4.3904187027399575e-03 0.0000000000000000e+00"
    rvecstr = "1.4279453733149436e+00 -6.5018755365706324e-02 6.9394619505977417e-02"
    tvecstr = "-9.3321723085362578e+02 8.1284728383272942e+00 5.5108407805855968e+03"
    imgs = np.fromstring(imgsstr, sep = " ").reshape(2).astype(int)
    cmat = np.fromstring(cmatstr, sep = " ").reshape(3,3)
    dvec = np.fromstring(dvecstr, sep = " ").reshape(-1,1)
    rvec = np.fromstring(rvecstr, sep = " ").reshape(3,1)
    tvec = np.fromstring(tvecstr, sep = " ").reshape(3,1)    
    cam = Camera(imgSize=imgs, cmat=cmat, dvec=dvec, rvec=rvec, tvec=tvec)
    print(cam)
    cam.plotCameraPyramid()
    
def test_camera_2():
    cam1 = Camera()
    cam1.imgSize=(3840,2160)
    cam1.cmat = np.array([5e3,0,1919.5, 0, 5e3, 1079.5, 0,0,1.]).reshape(3,3)
    cam1.dvec = np.array([-0.1,0,0,0]).flatten()
    pos1 = np.array([0, -300, 10.])
    aim1 = np.array([100,150,120])
    cam1.setRvecTvecByPosAim(pos1, aim1)
    ax = cam1.plotCameraPyramid()             
    
    cam2 = Camera()
    cam2.imgSize=(3840,2160)
    cam2.cmat = np.array([5e3,0,1919.5, 0, 5e3, 1079.5, 0,0,1.]).reshape(3,3)
    cam2.dvec = np.array([-0.1,0,0,0]).flatten()
    pos2 = np.array([250, -300, 10.])
    aim2 = np.array([100,150,120])
    cam2.setRvecTvecByPosAim(pos2, aim2)
    ax = cam2.plotCameraPyramid(axes=ax) 
    
    plot_quad_on_axes([ -50, 0, -50], [ 250, 0, -50],
                      [ 250, 0, 150], [ -50, 0, 150], 
                      color='blue', alpha=0.5, ax=ax)
    
if __name__ == '__main__':
    test_camera_2()
