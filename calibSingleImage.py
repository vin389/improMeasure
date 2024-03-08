import numpy as np
import cv2 as cv
from inputs import input2, input3
from imread2 import imread2
from readPoints import readPoints
from writePoints import writePoints
from writeCamera import writeCamera 
from removeNanPoints import removeNanPoints
from drawPoints import drawPoints


def calibSingleImage(img=None, objPoints=None, imgPoints=None, 
                     cmat=None, dvec=None,
                     flags=None, criteria=None, 
                     saveCamFile=None, savePrjFile=None, saveErrFile=None, 
                     saveImgFile=None):
    """
    Given image points, object points, and the image itself, this function
    estimates the intrinsic and extrinsic parameters of the camrea. 
    For example, 
        img = cv.imread('examples/calibSingleImage/IMG_0001.jpg')
        imgSize = (img.shape[1], img.shape[0])    
        objPoints = readPoints('examples/calibSingleImage/objPoints.csv')
        imgPoints = readPoints('examples/calibSingleImage/imgPoints.csv')
        guessFx = max(imgSize)
        guessCx = imgSize[0] * 0.5 - 0.5
        guessCy = imgSize[1] * 0.5 - 0.5
        cmat = np.array([[guessFx,0,guessCx],[0,guessFx,guessCy],[0,0,1]])
        dvec = np.zeros((1,14), dtype=float)
        flags = 205
        retVal, cmat, dvec, rvec, tvec = calibSingleImage(
            img=img, objPoints=objPoints, imgPoints=imgPoints,
            cmat=cmat, dvec=dvec, flags=flags, 
            saveCamFile = 'examples/calibSingleImage/camera_1.csv', 
            saveImgFile = 'examples/calibSingleImage/prj_IMG_0001.jpg', 
            savePrjFile = 'examples/calibSingleImage/prjPoints.csv',
            saveErrFile = 'examples/calibSingleImage/errPoints.csv')

    Parameters
    ----------
    img : gray or BGR 3-channel image
        calibration image. 
        If not given, this function interactives with the user. 
    objPoints : an N-by-3 float numpy, where N is # of points. 
        3D points of the calibration points. 
        If not given, this function interactives with the user. 
    imgPoints : an N-by-2 float numpy, where N is # of points.
        The image coordinates (in unit of pixel) of the calibration points.
        If not given, this function interactives with the user. 
    cmat : an 3-by-3 float numpy. 
        Initial guess of camera matrix. 
        If not given, it is assumed that the focal lengths are the 
        image size and the principle point is at the 
        center of the image. For example, if the image size is 
        6000 by 4000, the initial guess of focal lengths are 6000 and the 
        princlple point is at (2999.5, 1999.5).
    dvec : an 1 dimensional numpy , optional
        Initial guessed distortion coefficients.
        If not given, the initial guess of the dvec is a zero vector.
    flags : int, optional
        The flags for the calibration. See OpenCV documentation of 
        calibrateCamera() for definition of flags.
        Important: For non-planar calibration rigs the initial intrinsic 
        matrix must be specified (See OpenCV documentation). That is, the 
        flag CALIB_USE_INTRINSIC_GUESS (1) must be enabled, that is, the 
        flags must be an odd (cannot be zero or any other even)
        Typical values are: 
            0: OpenCV default (calibrates fx, fy, cx, cy, k1, k2, p1, p2, k3)
            16385: (calibrates fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6)
            24705: (calibrates fx, fy, cx, cy, k1, k2, p1, p2,   , k4, k5)
            129: (calibrates fx, fy, cx, cy, k1, k2, p1, p2)
            193: (calibrates fx, fy, cx, cy, k1, p1, p2)
            201: (calibrates fx, fy, cx, cy, k1)
            205: (calibrates fx, fy, k1)
            207: (calibrates fx, k1)
            To remove CALIB_USE_INTRINSIC_GUESS (if you do not want to guess
            the camera matrix), use the above number minus 1. For example,
            give 206 (i.e., 207 - 1) to calibrate fx and k1 without using 
            initial guess camera matrix. 
        If not given, this function interactives with the user. 
          cv.CALIB_USE_INTRINSIC_GUESS = 0x00001,
          cv.CALIB_FIX_ASPECT_RATIO = 0x00002,
          cv.CALIB_FIX_PRINCIPAL_POINT = 0x00004,
          cv.CALIB_ZERO_TANGENT_DIST = 0x00008,
          cv.CALIB_FIX_FOCAL_LENGTH = 0x00010,
          cv.CALIB_FIX_K1 = 0x00020,
          cv.CALIB_FIX_K2 = 0x00040,
          cv.CALIB_FIX_K3 = 0x00080,
          cv.CALIB_FIX_K4 = 0x00800,
          cv.CALIB_FIX_K5 = 0x01000,
          cv.CALIB_FIX_K6 = 0x02000,
          cv.CALIB_RATIONAL_MODEL = 0x04000,
          cv.CALIB_THIN_PRISM_MODEL = 0x08000,
          cv.CALIB_FIX_S1_S2_S3_S4 = 0x10000,
          cv.CALIB_TILTED_MODEL = 0x40000,
          cv.CALIB_FIX_TAUX_TAUY = 0x80000,
          cv.CALIB_USE_QR = 0x100000,
          cv.CALIB_FIX_TANGENT_DIST = 0x200000,
          cv.CALIB_FIX_INTRINSIC = 0x00100,
          cv.CALIB_SAME_FOCAL_LENGTH = 0x00200,
          cv.CALIB_ZERO_DISPARITY = 0x00400,
          cv.CALIB_USE_LU = (1 << 17),
          cv.CALIB_USE_EXTRINSIC_GUESS = (1 << 22)
    criteria : tuple (int,int,float), optional
        The criteria for the calibration. For example, 
        (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03)
        See OpenCV documentation of 
        calibrateCamera() for definition of flags. 
        If not given, this function uses OpenCV default values. 
    saveCamFile : str, optional
        The output file of camera parameters (in csv format)
        (rvec(length of 3), tvec(3), cmat(9), dvec(4,5,8,12,or 14))
        If the file name is "cout" or "stdout" parameters will be printed 
        on the screen.
        If the file name is "" parameters will not be written to file.
        If not given, this function will ask user. 
    savePrjFIle : str, optional
        The output file of projected points (in csv format)
        If the file name is "cout" or "stdout" the projected points will be 
        printed on the screen.
        If the file name is "" the projected points will not be written 
        to file.
        If not given, this function will ask user. 
    saveErrFIle : str, optional
        The output file of error of projected points (in csv format)
        That is, projected - actual. 
        If the file name is "cout" or "stdout" the error will be 
        printed on the screen.
        If the file name is "" the error will not be written 
        to file.
        If not given, this function will ask user. 
    saveImgFile : str, optional
        The output image file of calibration points
        The user-picked points are marked with blue crosses.
        The projected points are makred with red crosses.
        If not given, this function will ask user. 
    Returns
    -------
    errProj : float
        errProj average projetion errors 
    rvec : np.ndarray (shape: 3)
        rvec rotational vector of the extrinsic parameters
    tvec : np.ndarray (shape: 3)
        tvec translational vector of the extrinsic parameters
    cmat : np.ndarray (shape: (3,3))
        cmat the camera matrix 
    dvec : np.ndarray (shape: n, n can be 4, 5, 8, 12, or 14)
        dvec the distortion coefficients (k1, k2, p1, p2[, k3, k4, k5, k6[
            , s1, s2, s3, s4[, taux, tauy]]]) 

    """
    # get image and image size
    if (type(img) == type(None)):
        print("# Enter image for calibration:")
        img = imread2()
        if type(img) == type(None):
            print("# Error: calibSingleImage(): imread2() returns None.")
            return []
    imgSize = (img.shape[1], img.shape[0])
#    print("# calibSingleImage(): Image file size is %dx%d (wxh)" % (imgSize[0], imgSize[1]))

    # image points
    if (type(imgPoints) == type(None)):
        print("# Enter image points (2D) for calibration:")
        print("#  Suggestion: Give more than 15 calibration points.")
        print("#  Suggestion: Calibration points should cover the area of"
              " your measurement region.")
        imgPoints = readPoints()

    # object points
    if (type(objPoints) == type(None)):
        print()
        print("# Enter object points (3D) for calibration:")
        print("#  Suggestion: Give more than 15 calibration points.")
        print("#  Suggestion: Calibration points should cover the area of"
              " your measurement region.")
        objPoints = readPoints()
    # default flag
    if (type(flags) == type(None)):
        flagsStr = ['CALIB_USE_INTRINSIC_GUESS',
            'CALIB_FIX_ASPECT_RATIO',
            'CALIB_FIX_PRINCIPAL_POINT',
            'CALIB_ZERO_TANGENT_DIST',
            'CALIB_FIX_FOCAL_LENGTH',
            'CALIB_FIX_K1',
            'CALIB_FIX_K2',
            'CALIB_FIX_K3',
            'CALIB_FIX_K4',
            'CALIB_FIX_K5',
            'CALIB_FIX_K6',
            'CALIB_RATIONAL_MODEL',
            'CALIB_THIN_PRISM_MODEL',
            'CALIB_FIX_S1_S2_S3_S4',
            'CALIB_TILTED_MODEL',
            'CALIB_FIX_TAUX_TAUY',
            'CALIB_USE_QR',
            'CALIB_FIX_TANGENT_DIST',
            'CALIB_FIX_INTRINSIC',
            'CALIB_SAME_FOCAL_LENGTH',
            'CALIB_ZERO_DISPARITY',
            'CALIB_USE_LU',
            'CALIB_USE_EXTRINSIC_GUESS']
        flags = 0
        print("# Enter calibration flags one by one (0 or no, 1 for yes).")
        for i in range(len(flagsStr)):
            print("#  Do you want to use flag %s? " % (flagsStr[i]))
            uInput = input3('',dtype=int,min=0,max=1)
            flags = flags | (uInput * eval('cv.' + flagsStr[i]))
        print("# calibSingleImage(): The calibration flags is %d" % (flags))
    # camera matrix
    if type(cmat) == type(None):
        # if cmat is not given, it estimates cmat as a perfect FOV 
        # of roughly 53 deg. or 35 mm, and eventually these parameters 
        # would be adjusted in calibration later as long as proper flags
        # are given.     
        guessFx = max(imgSize)
        guessFy = guessFx
        guessCx = imgSize[0] * .5 - .5
        guessCy = imgSize[1] * .5 - .5
        guessCmat = np.array([[guessFx, 0, guessCx],
                              [0, guessFy, guessCy],[0,0,1.]])
        cmat = guessCmat     
    # distortion coefficients
    if type(dvec) == type(None):
        # estimate dvec
        guessDvec = np.zeros((1,14))
        dvec = guessDvec
    # remove nan points
    objPoints, imgPoints = removeNanPoints(objPoints, imgPoints)  
    # convert to format for cv.calibrateCamera
    imgPoints2f = np.array(imgPoints, dtype=np.float32)
    imgPoints2f = imgPoints2f.reshape((1,-1,2))
    objPoints3f = np.array(objPoints, dtype=np.float32)
    objPoints3f = objPoints3f.reshape((1,-1,3))
    # call function cv.calibrateCamera()
    retVal, cmat, dvec, rvec, tvec = cv.calibrateCamera(
        objPoints3f, imgPoints2f, imageSize=imgSize, 
        cameraMatrix=cmat, distCoeffs=dvec, flags=flags)
    if type(rvec) == tuple:
        rvec = rvec[0]
    if type(tvec) == tuple:
        tvec = tvec[0]    
    # write camera parameters to file
    if type(saveCamFile) == type(None):
        print("# Enter file name that you want to save camera parameters:")
        print("#   You can enter cout or stdout to only print on screen.")
        print("#   or enter dot (.) to skip saving.")
        saveCamFile = input2()
    if (saveCamFile != '.'):
        writeCamera(saveCamFile, rvec, tvec, cmat, dvec)
    # calculate and write projected points to file
    if type(savePrjFile) == type(None):
        print("# Enter file name that you want to save projected points:")
        print("#   or enter dot (.) to skip saving.")
        savePrjFile = input2()
    if (savePrjFile != '.'):
        prjPoints, jacobian = cv.projectPoints(objPoints, rvec, tvec, cmat, dvec)
        prjPoints = prjPoints.reshape(-1,2)
        writePoints(savePrjFile, prjPoints, 'Projected points')
    # calculate and write projected errors to file
    if type(saveErrFile) == type(None):
        print("# Enter file name that you want to save projected errors:")
        print("#   or enter dot (.) to skip saving.")
        saveErrFile = input2()
    if (saveErrFile != '.'):
        errPoints = prjPoints - imgPoints    
        writePoints(saveErrFile, errPoints, 'Projected errors (proj - actual)')
    # draw calibration points and projected points on the image
    if type(saveImgFile) == type(None):
        print("# Enter image file name that you want to draw points:")
        print("#   You can enter dot (.) to skip saving.")
        saveImgFile = input2()
    if (saveImgFile != '.'):
        img = drawPoints(img, imgPoints, color=[0,0,0], markerSize=20, 
                         thickness=5, savefile=".")
        img = drawPoints(img, imgPoints, color=[255,0,0], markerSize=18, 
                         thickness=3, savefile=".")
        img = drawPoints(img, prjPoints, color=[0,0,0], markerSize=20, 
                         thickness=5, savefile=".")
        img = drawPoints(img, prjPoints, color=[0,0,255], markerSize=18, 
                         thickness=3, savefile=saveImgFile)
    # return 
    return retVal,rvec,tvec,cmat,dvec


def test_calibSingleImage():
    img = cv.imread('examples/calibSingleImage/IMG_0001.jpg')
    imgSize = (img.shape[1], img.shape[0])    
    objPoints = readPoints('examples/calibSingleImage/objPoints.csv')
    imgPoints = readPoints('examples/calibSingleImage/imgPoints.csv')
    guessFx = max(imgSize)
    guessCx = imgSize[0] * 0.5 - 0.5
    guessCy = imgSize[1] * 0.5 - 0.5
    cmat = np.array([[guessFx,0,guessCx],[0,guessFx,guessCy],[0,0,1]])
    dvec = np.zeros((1,14), dtype=float)
    flags = 205
    retVal, cmat, dvec, rvec, tvec = calibSingleImage(
        img=img, objPoints=objPoints, imgPoints=imgPoints,
        cmat=cmat, dvec=dvec, flags=flags, 
        saveCamFile = 'examples/calibSingleImage/camera_1.csv', 
        saveImgFile = 'examples/calibSingleImage/prj_IMG_0001.jpg', 
        savePrjFile = 'examples/calibSingleImage/prjPoints.csv',
        saveErrFile = 'examples/calibSingleImage/errPoints.csv')
#test_calibSingleImage()


                 


