import cv2 as cv
import numpy as np 
from inputs import input2, input3
from writePoints import writePoints
from writeCamera import writeCamera 
from drawPoints import drawPoints
from createFileList import createFileList


def calibChessboard2(fileList=None, patternSize=None, cellSize=None, 
                    cmat=None, dvec=None,
                    flags=None, criteria=None, 
                    saveCamFile=None, 
                    saveObjPointsFile=None, saveImgPointsFile=None,
                    savePrjFile=None, saveErrFile=None, 
                    saveImgFile=None):
    """
    This function runs chessboard calibration given chessboard images.

    Parameters
    ----------
    fileList : list of strings, optional
        List of calibration file names (full-path). The default is None.
        For example, ['/calib/IMG_0001.JPG', '/calib/IMG_0002.JPG', ... ]
    patternSize : Tuple, optional
        Number of inner corners per a chessboard row and column 
        (points_per_row,points_per_column). The default is None.
        For example, (7, 7) for a standard chessboard (not (8,8))
    cellSize : Tuple, optional
        Physical size of a cell of chessboard. The default is None.
        For example, (50.8, 50.8) 
    cmat : 3-by-3 numpy float matrix, optional
        Camera matrix. The default is None.
        That is, [fx 0 cx; 0 fy cy; 0 0 1]
        For example, np.array([[1800, 0, 960], [0, 1800, 540], [0, 0, 1.]])
    dvec : numpy float row vector, optional
        Distortion coefficients. The default is None.
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
    saveObjPointsFile : str, optional
        The output file of object points (in csv format)
        If the file name is "cout" or "stdout" the projected points will be 
        printed on the screen.
        If the file name is "" the projected points will not be written 
        to file.
        If not given, this function will ask user. 
    saveImgPointsFile : str, optional
        The output file of image points (the detected corners) (in csv format)
        If the file name is "cout" or "stdout" the projected points will be 
        printed on the screen.
        If the file name is "" the projected points will not be written 
        to file.
        If not given, this function will ask user. 
    savePrjFile : str, optional
        The output file of projected points (in csv format)
        If the file name is "cout" or "stdout" the projected points will be 
        printed on the screen.
        If the file name is "" the projected points will not be written 
        to file.
        If not given, this function will ask user. 
    saveErrFile : str, optional
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
    objPoints : np.ndarray(shape: (nPics, patternSize[0] * patternSize[1], 3))
    imgPoints : np.ndarray(shape: (nPics, patternSize[0] * patternSize[1], 2))
    prjPoints : np.ndarray(shape: (nPics, patternSize[0] * patternSize[1], 2))
        prjPoints the projected points on the images
    errEveryPoints : np.ndarray(shape: (nPics, patternSize[0] * patternSize[1], 2))
        errPoints the error: prjPoints - imgPoints
        
    Example
    -------
    retVal, cmat, dvec, rvec, tvec = calibChessboard(
        fileList=['/pics/IMG0.BMP', '/pics/IMG2.BMP', '/pics/IMG3.BMP'] 
        patternSize=(7, 7),
        cellSize=(50.8, 50.8),
        flags=201,
        saveCamFile='/pics/calib/camCalibParameters.csv',
        saveObjPointsFile='/pics/calib/objPoints.csv',
        saveImgPointsFile='/pics/calib/imgPoints.csv',
        savePrjFile='/pics/calib/camCalibPrjPoints.csv',
        saveErrFile='/pics/calib/camCalibPrjErrors.csv',
        saveImgFile='/pics/calib/camCalibPrjImage.csv')

    """
    # fileList
    if (type(fileList) == type(None)):
        print("# The calibChessboard() needs a file list of images. ")
        print("# For calibChessboard(), you can enter:")
        print("#   examples/calibChessboard/G*.JPG")
        fileList = createFileList()
        print("# The calibration images are:")
        print("# ", fileList)
    # patternSize
    if (type(patternSize) == type(None)):
        print("# The calibChessboard() needs pattern size of chessboard.")
        print("#   Number of inner corners per a chessboard row and column ")
        print("#   (points_per_row,points_per_column). The default is None.")
        print("#   for example, (7, 7) for a standard chessboard (not (8,8))")
        print("# You can enter (7,7) for a standard chessboard.")
        print("# For calibChessboard() example, you can enter:")
        print("#   (7, 12)")
        patternSize = eval(input2())
        if (type(patternSize) != tuple or len(patternSize) != 2):
            print("# Error: patternSize needs to be a 2-integer tuple.")
            return
        print("# Number of inner corners is ", patternSize)
    # cellSize
    if (type(cellSize) == type(None)):
        print("# The calibChessboard() needs cell size of a cell of the "
              "chessboard.")
        print("#   It is the physical size of a cell of chessboard. ")
        print("# For example, (50.8, 50.8)")
        print("# The unit can be mm, inch, m, or any length unit you prefer")
        print("#   but needs to be consistent through you entire analysis.")
        print("# For calibChessboard() example, you can enter:")
        print("#   (21.31, 21.35)")
        cellSize = eval(input2())
        if (type(cellSize) != tuple or len(cellSize) != 2):
            print("# Error: cellSize needs to be a 2-integer tuple.")
            return
        print("# The cell size is ", cellSize)
    # imgSize, get imgSize by reading the first image in fileList
    img = cv.imread(fileList[0])
    if (type(img) != type(None) and img.shape[0] > 0):
        imgSize = (img.shape[1], img.shape[0])   
    # cmat
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
    # flags
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
        print("# calibChessboard(): The calibration flags is %d" % (flags))
    # write camera parameters to file
    if type(saveCamFile) == type(None):
        print("# Enter file name that you want to save camera parameters:")
        print("#   You can enter cout or stdout to only print on screen.")
        print("#   or enter dot (.) to skip saving.")
        saveCamFile = input2()
    if type(saveObjPointsFile) == type(None):
        print("# Enter file name that you want to save object points of "
              "chessboard corners:")
        print("#   or enter dot (.) to skip saving.")
        saveObjPointsFile = input2()
    if type(saveImgPointsFile) == type(None):
        print("# Enter file name that you want to save image points of "
              "chessboard corners:")
        print("#   or enter dot (.) to skip saving.")
        saveImgPointsFile = input2()
    if type(savePrjFile) == type(None):
        print("# Enter file name that you want to save projected points:")
        print("#   or enter dot (.) to skip saving.")
        savePrjFile = input2()
    if type(saveErrFile) == type(None):
        print("# Enter file name that you want to save projected errors:")
        print("#   or enter dot (.) to skip saving.")
        saveErrFile = input2()
    if type(saveImgFile) == type(None):
        print("# Enter image file name that you want to draw points:")
        print("#   You can enter dot (.) to skip saving.")
        saveImgFile = input2()
    # allocate main arrays        
    nFiles = len(fileList)
    imgPoints2f = np.zeros((nFiles * patternSize[0] * patternSize[1],2), 
                           dtype=np.float32)
    imgPoints2f = imgPoints2f.reshape((nFiles,-1,2))
    prjPoints2f = imgPoints2f.copy()
    objPoints3f = np.zeros((nFiles * patternSize[0] * patternSize[1],3), 
                           dtype=np.float32)
    objPoints3f = objPoints3f.reshape((nFiles,-1,3))
    nFound = 0
    # find chessboard corners
    for i in range(nFiles):
        filename = fileList[i]
        img = cv.imread(filename, cv.IMREAD_GRAYSCALE)
        if (type(img) != type(None) and img.shape[0] > 0 
            and img.shape[1] > 0):
            imgSize = (img.shape[1], img.shape[0])
        found, ptsThis = cv.findChessboardCorners(img, patternSize)
        if found==True:
            # to subpixel accuracy
            # (recent version of findChessboardCorners has been subpixel based
            #  and we do not need to do subpix analysis additionally) 
#            winSize = (5, 5)
#            zeroZone = (-1, -1)
#            criteria = (cv.TERM_CRITERIA_EPS + cv.TermCriteria_COUNT, 40, 0.01)
#            ptsThis = cv.cornerSubPix(img, ptsThis, winSize, 
#                                      zeroZone, criteria)
            imgPoints2f[nFound,:,:] = ptsThis[:,0,:]
            print("# Calibration image %d corners detection: all found." 
                  % (i + 1))
            for iy in range(patternSize[1]):
                for ix in range(patternSize[0]):
                    objPoints3f[nFound, ix + iy * patternSize[0], 0] \
                        = ix * cellSize[0]
                    objPoints3f[nFound, ix + iy * patternSize[0], 1] \
                        = iy * cellSize[1]
                    objPoints3f[nFound, ix + iy * patternSize[0], 2] = 0.0
            nFound += 1
        else:
            print("# Error: Calibration image %d corners detection:"
                  " not all found." % (i + 1))
            print("# Error: You need to remove the file from calibration image: "
                  "%s" % (filename))
            return 
    # calibrate camera
    retVal, cmat, dvec, rvec, tvec = cv.calibrateCamera(
        objPoints3f, imgPoints2f, imageSize=imgSize, 
        cameraMatrix=cmat, distCoeffs=dvec, flags=flags)
    # only keep the rvec of the first calibration image
    #if type(rvec) == tuple:
    #    rvec = rvec[0]
    #if type(tvec) == tuple:
    #    tvec = tvec[0]
    # write camera parameters to file
    if type(saveCamFile) == type(None):
        print("# Enter file name that you want to save camera parameters:")
        print("#   You can enter cout or stdout to only print on screen.")
        print("#   or enter dot (.) to skip saving.")
        saveCamFile = input2()
    if (saveCamFile != '.'):
        writeCamera(saveCamFile, rvec[0], tvec[0], cmat, dvec)
        # write a camera file to each calibration image 
        # if the saveCamFile is MyCalib.csv, the output files would be 
        # MyCalib_calib_001.csv, MyCalib_calib_002.csv, ... and so on.
        for i in range(nFiles):
            saveCamFile_i = saveCamFile[0:-4] + "_calib_%03d" % (i + 1) \
                + saveCamFile[-4:] 
            writeCamera(saveCamFile_i, rvec[i], tvec[i], cmat, dvec)
    # write image (corner) points files
    if type(saveImgPointsFile) == type(None):
        print("# Enter file name that you want to save image points:")
        print("#   or enter dot (.) to skip saving.")
        saveImgPointsFile = input2()
    if (saveImgPointsFile != '.'):
        writePoints(saveImgPointsFile, imgPoints2f[0,:,:], 
                    'Chessboard Corners')
        # write image point file to each calibration image 
        # if the saveObjPointsFile is MyImgPoints.csv, the output files would 
        # be 
        # MyImgPoints_imgpts_001.csv, MyImgPoints_imgpts_002.csv, ... and so
        # on.
        for i in range(nFiles):
            saveImgPointsFile_i = saveImgPointsFile[0:-4] + "_imgpts_%03d" \
                % (i + 1) + saveImgPointsFile[-4:] 
            writePoints(saveImgPointsFile_i, imgPoints2f[i,:,:], 
                        'Chessboard Corners')
    # write object points files
    if type(saveObjPointsFile) == type(None):
        print("# Enter file name that you want to save object points:")
        print("#   or enter dot (.) to skip saving.")
        saveObjPointsFile = input2()
    if (saveObjPointsFile != '.'):
        writePoints(saveObjPointsFile, objPoints3f[0,:,:], 
                    'Chessboard Object Points')    
    # calculate projected points
    objPoints = np.array(objPoints3f[0,:,:], dtype=float)
    for i in range(nFiles):
        prjPoints, jacobian = cv.projectPoints(objPoints, 
                                               rvec[i], tvec[i], cmat, dvec)
        prjPoints = prjPoints.reshape(-1,2)
        prjPoints2f[i,:,:] = prjPoints[:,:]
    # calculate and write projected points to file
    if type(savePrjFile) == type(None):
        print("# Enter file name that you want to save projected points:")
        print("#   or enter dot (.) to skip saving.")
        savePrjFile = input2()
    if (savePrjFile != '.'):
        writePoints(savePrjFile, prjPoints2f[0,:,:], 'Projected points')
        # write a projected point file to each calibration image 
        # if the savePrjFile is MyProj.csv, the output files would be 
        # MyProj_proj_001.csv, MyProj_proj_002.csv, ... and so on.
        for i in range(nFiles):
            savePrjFile_i = savePrjFile[0:-4] + "_proj_%03d" % (i + 1) \
                + savePrjFile[-4:] 
            writePoints(savePrjFile_i, prjPoints2f[i,:,:], 'Projected points')
    # calculate and write projected errors to file
    if type(saveErrFile) == type(None):
        print("# Enter file name that you want to save projected errors:")
        print("#   or enter dot (.) to skip saving.")
        saveErrFile = input2()
    if (saveErrFile != '.'):
        errPoints = prjPoints2f[0,:,:] - imgPoints2f[0,:,:]
        writePoints(saveErrFile, errPoints, 'Projected errors (proj - actual)')
        # write a project err point file to each calibration image 
        # if the saveErrFile is MyErr.csv, the output files would be 
        # MyErr_err_001.csv, MyErr_err_002.csv, ... and so on.
        for i in range(nFiles):
            errPoints = prjPoints2f[i,:,:] - imgPoints2f[i,:,:]
            saveErrFile_i = saveErrFile[0:-4] + "_err_%03d" % (i + 1) \
                + saveErrFile[-4:] 
            writePoints(saveErrFile_i, errPoints, 
                        'Projected errors (proj - actual)')
    # draw calibration points and projected points on the image
    if type(saveImgFile) == type(None):
        print("# Enter image file name that you want to draw points:")
        print("#   You can enter dot (.) to skip saving.")
        saveImgFile = input2()
    if (saveImgFile != '.'):
        img = cv.imread(fileList[0])
        img = drawPoints(img, imgPoints2f[0,:,:], color=[0,0,0], 
                         markerSize=20, thickness=5, savefile=".")
        img = drawPoints(img, imgPoints2f[0,:,:], color=[255,0,0], 
                         markerSize=18, thickness=3, savefile=".")
        img = drawPoints(img, prjPoints2f[0,:,:], color=[0,0,0], 
                         markerSize=20, thickness=5, savefile=".")
        img = drawPoints(img, prjPoints2f[0,:,:], color=[0,0,255], 
                         markerSize=18, thickness=3, savefile=saveImgFile)
        # write a proj image file to each calibration image 
        # if the saveImgFile is MyImg.JPG, the output files would be 
        # MyImg_img_001.JPG, MyImg_img_002.JPG, ... and so on.
        for i in range(nFiles):
            saveImgFile_i = saveImgFile[0:-4] + "_prjImg_%03d" % (i + 1) \
                + saveImgFile[-4:] 
            img = cv.imread(fileList[i])
            img = drawPoints(img, imgPoints2f[i,:,:], color=[0,0,0], 
                             markerSize=20, thickness=5, savefile=".")
            img = drawPoints(img, imgPoints2f[i,:,:], color=[255,0,0], 
                             markerSize=18, thickness=3, savefile=".")
            img = drawPoints(img, prjPoints2f[i,:,:], color=[0,0,0], 
                             markerSize=20, thickness=5, savefile=".")
            img = drawPoints(img, prjPoints2f[i,:,:], color=[0,0,255], 
                             markerSize=18, thickness=3, 
                             savefile=saveImgFile_i)
    # return 
    return retVal,rvec,tvec,cmat,dvec


def test_calibChessboard_example():
    # fileList=None, patternSize=None, cellSize=None, 
    #                     cmat=None, dvec=None,
    #                     flags=None, criteria=None, 
    #                     saveCamFile=None, savePrjFile=None, saveErrFile=None, 
    #                     saveImgFile=None
    pwd = "D:\\yuansen\\ImPro\\impropy\\impropy\\examples\\calibChessboard\\"
    myFileList = createFileList(pwd + "G00*.JPG", ".") # , or
    # myFileList = [pwd + "G0016456.JPG", 
    #               pwd + "G0016461.JPG", 
    #               pwd + "G0016466.JPG"]
    myPatternSize = (7, 12)
    myCellSize = (21.31, 21.35)
    # Not giving cmat and dvec leads to an initial guess of 53-deg FOV cmat.
    # with zero distortion. Eventually they will be refined in calibration.
    myFlags = 201 # only calibrates fx, fy, cx, cy, and k1.
        # 0: OpenCV default (calibrates fx, fy, cx, cy, k1, k2, p1, p2, k3)
        # 16385: (calibrates fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6)
        # 24705: (calibrates fx, fy, cx, cy, k1, k2, p1, p2,   , k4, k5)
        # 129: (calibrates fx, fy, cx, cy, k1, k2, p1, p2)
        # 193: (calibrates fx, fy, cx, cy, k1, p1, p2)
        # 201: (calibrates fx, fy, cx, cy, k1)
        # 205: (calibrates fx, fy, k1)
        # 207: (calibrates fx, k1)
    # criteria = (cv.TERM_CRITERIA_EPS + cv.TermCriteria_COUNT, 40, 0.001)
    mySaveCamFile = pwd + "testOutput/" + "cameraParameters.csv"
    mySaveImgPointsFile = pwd + "testOutput/" \
                        + "chessboardCornersImgPoints.csv"
    mySaveObjPointsFile = pwd + "testOutput/" \
                        + "chessboardCornersObjPoints.csv"
    mySavePrjFile = pwd + "testOutput/" + "projectedPoints.csv"
    mySaveErrFile = pwd + "testOutput/" + "projectionErrors.csv"
    mySaveImgFile = pwd + "testOutput/" + "projectionImage.JPG"
    # run calibChessboard, optionally not giving cmat, dvec, and criteria
    retVal, cmat, dvec, rvec, tvec = calibChessboard(
        fileList=myFileList, 
        patternSize=myPatternSize,
        cellSize=myCellSize,
        flags=myFlags,
        saveCamFile=mySaveCamFile,
        saveImgPointsFile=mySaveImgPointsFile,
        saveObjPointsFile=mySaveObjPointsFile,
        savePrjFile=mySavePrjFile,
        saveErrFile=mySaveErrFile,
        saveImgFile=mySaveImgFile)
#test_calibSingleImage()


def test_calibChessboard_brb1():
    # fileList=None, patternSize=None, cellSize=None, 
    #                     cmat=None, dvec=None,
    #                     flags=None, criteria=None, 
    #                     saveCamFile=None, savePrjFile=None, saveErrFile=None, 
    #                     saveImgFile=None
    pwd = "D:\\ExpDataSamples\\20220500_Brb\\brb1\\brb1_cam5_neck_calib\\"
    myFileList = createFileList(pwd + "G00*.JPG", ".") # , or
    # myFileList = [pwd + "G0016456.JPG", 
    #               pwd + "G0016461.JPG", 
    #               pwd + "G0016466.JPG"]
    myPatternSize = (7, 12)
    myCellSize = (21.31, 21.35)
    # Not giving cmat and dvec leads to an initial guess of 53-deg FOV cmat.
    # with zero distortion. Eventually they will be refined in calibration.
    myFlags = 201 # only calibrates fx, fy, cx, cy, and k1.
        # 0: OpenCV default (calibrates fx, fy, cx, cy, k1, k2, p1, p2, k3)
        # 16385: (calibrates fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6)
        # 24705: (calibrates fx, fy, cx, cy, k1, k2, p1, p2,   , k4, k5)
        # 129: (calibrates fx, fy, cx, cy, k1, k2, p1, p2)
        # 193: (calibrates fx, fy, cx, cy, k1, p1, p2)
        # 201: (calibrates fx, fy, cx, cy, k1)
        # 205: (calibrates fx, fy, k1)
        # 207: (calibrates fx, k1)
    # criteria = (cv.TERM_CRITERIA_EPS + cv.TermCriteria_COUNT, 40, 0.001)
    mySaveCamFile = pwd + "brb1_cam5_parameters.csv"
    mySaveImgPointsFile = pwd + "brb1_cam5_imgPoints.csv"
    mySaveObjPointsFile = pwd + "brb1_cam5_objPoints.csv"
    mySavePrjFile = pwd + "brb1_cam5_proj_test.csv"
    mySaveErrFile = pwd + "brb1_cam5_err_test.csv"
    mySaveImgFile = pwd + "brb1_cam5_proj_img.JPG"
    # run calibChessboard, optionally not giving cmat, dvec, and criteria
    retVal, cmat, dvec, rvec, tvec = calibChessboard(
        fileList=myFileList, 
        patternSize=myPatternSize,
        cellSize=myCellSize,
        flags=myFlags,
        saveCamFile=mySaveCamFile,
        saveImgPointsFile=mySaveImgPointsFile,
        saveObjPointsFile=mySaveObjPointsFile,
        savePrjFile=mySavePrjFile,
        saveErrFile=mySaveErrFile,
        saveImgFile=mySaveImgFile)


#test_calibSingleImage()
def chessboardPts3d(nRows=7, nCols=7, cellSize=1.0):
    pts3d = np.zeros((nRows * nCols, 3), dtype=float)
    for i in range(nRows):
        for j in range(nCols):
            pts3d[i * nCols + j, 0] = i * cellSize
            pts3d[i * nCols + j, 1] = j * cellSize
            pts3d[i * nCols + j, 2] = 0.0
    return pts3d


def chessboardPts2d(nRows=7, nCols=7, cellSize=1.0):
    pts2d = np.zeros((nRows * nCols, 2), dtype=float)
    for i in range(nRows):
        for j in range(nCols):
            pts2d[i * nCols + j, 0] = i * cellSize
            pts2d[i * nCols + j, 1] = j * cellSize
    return pts2d