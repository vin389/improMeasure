import cv2 as cv
import numpy as np
import glob 
from inputs import input2, input3
from writePoints import writePoints
from readCamera import readCamera
from writeCamera import writeCamera
from drawPoints import drawPoints
from createFileList import createFileList
from calibChessboard import calibChessboard

# This file contains the following functions
#   calibChessboardStereo()
#   calibStereo_test()
#   calibStereo_test_brb_example()


def calibChessboardStereo(fileListL=None,
                          fileListR=None,
                          patternSize=None,
                          cellSize=None,
                          flagsL=None,
                          flagsR=None,
                          outputDir=None):
    """
    Runs stereo (two-camera) calibration and returns intrinsic and extrinsic
    parameters of both cameras, 
    given chessboard images, chessboard 
    pattern size (number of cells), cell size (physical size of a cell), 
    calibration flags, and the output directory. 

    Parameters
    ----------
    fileListL : string, optional
        Defines the files of the chessboard photos taken by left camera.
        For examples,
            "c:/dir1/*.PNG"
            "["c:/dir1/C1.PNG", "c:/dir1/C2.PNG", "c:/dir1/C3.PNG"]"
        The default is None. If it is NONE, this function asks 
        the user throught terminal.
    fileListR : string, optional
        Defines the files of the chessboard photos taken by right camera.
        For examples,
            "c:/dir2/*.PNG"
            "["c:/dir2/C1.PNG", "c:/dir2/C2.PNG", "c:/dir2/C3.PNG"]"
        The default is None. 
        If it is NONE, this function asks the user throught terminal.
    patternSize : tuple or list of two integers, optional
        Defines the number of cells of the chessboard along local x and y axes.
        (num_cells_along_local_x, num_cells_along_local_y)
        For example:
        (9, 12)
        [9, 12]
        np.array((9, 12), dtype=int)
        If it is NONE, this function asks the user throught terminal.
    cellSize : tuple or list of two floats, optional
        Defines the physical size of a cell of the chessboard along local x 
        and y axes. (cell_size_along_local_x, cell_size_along_local_y)
        For example:
        (25.4, 25.4)
        [23.13, 23.13]
        np.array([25.4, 25.4], dtype=float)
        If it is NONE, this function asks the user throught terminal.
    flagsL : TYPE, optional
        The flags for the calibration. See OpenCV documentation of 
        calibrateCamera() about the definition of flags.
        Important: For non-planar calibration rigs the initial intrinsic 
        matrix must be specified (See OpenCV documentation). 
        That is, if non-planar calibration rigs is used, the flag 
        CALIB_USE_INTRINSIC_GUESS (1) must be enabled (i.e., flags must be 
        odd) and initial guess of cmat and dvec must be given.
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
    flagsR : TYPE, optional
        See flagsL
    outputDir : string, optional
        Defines the directory where this function writes the output files.

    Returns
    -------
    TYPE
        DESCRIPTION.
        returnObject[0] : TYPE
            The return value .
    zeroArray : TYPE
        DESCRIPTION.
    zeroArray : TYPE
        DESCRIPTION.
    zeroArray : TYPE
        DESCRIPTION.
    zeroArray : TYPE
        DESCRIPTION.
    zeroArray : TYPE
        DESCRIPTION.

    Example:
    ----------
    fileListL = "D:/yuansen/ImPro/2022_MOST/guiTests/improMeasure/" +\
                "/examples/2022brb2c56/calibration_cboard_1/*.JPG"
    fileListR = "D:/yuansen/ImPro/2022_MOST/guiTests/improMeasure/" +\
                "/examples/2022brb2c56/calibration_cboard_2/*.JPG"
    patternSize = (7, 12)
    cellSize = (21.31, 21.35)
    flagsL = 193
    flagsR = 193
    outputDir = "D:\\yuansen\\ImPro\\2022_MOST\\guiTests\\improMeasure\\" +\
                "examples\\2022brb2c56\\calibration_cboard_2\\calib_193"
    retVal, cmatL, dvecL, cmatR, dvecR, R, T, E, F = \
        calibChessboardStereo(fileListL, fileListR, \
        patternSize, cellSize, flagsL, flagsR, outputDir)



    """
    # fileList
    if (type(fileListL) == type(None)):
        print("# The calibChessboardStereo() needs the file list of LEFT"
              " calibration images. ")
        print("# For calibChessboardStereo(), you can enter:")
        print("#   examples/calibChessboardStereo/cam5_calib/G*.JPG")
        fileListL = createFileList()
        print("# The left calibration images are:")
        print("# ", fileListL)
    if (type(fileListR) == type(None)):
        print("# The calibChessboardStereo() needs the file list of RIGHT"
              " calibration images. ")
        print("# For calibChessboardStereo(), you can enter:")
        print("#   examples/calibChessboardStereo/cam6_calib/G*.JPG")
        fileListR = createFileList()
        print("# The right calibration images are:")
        print("# ", fileListR)
    if (type(fileListL) == str):
        fileListL = createFileList(fileListL, ".")
    if (type(fileListR) == str):
        fileListR = createFileList(fileListR, ".")
    if (len(fileListL) != len(fileListR)):
        print("# Error: calibChessboardStereo(): LEFT and RIGHT must have" \
              "the same numbers of calibration images, but now LEFT has %d" \
              " and RIGHT has %d images." % (len(fileListL), len(fileListR)))
        zeroArray = np.zeros((0))
        return np.nan, zeroArray, zeroArray, zeroArray, zeroArray, zeroArray,\
               zeroArray
    # pattern size and cell size of the chessboard
    if (type(patternSize) == type(None)):
        print("# The calibChessboardStereo() needs pattern size of"
              " the chessboard.")
        print("#   Number of inner corners per a chessboard row and column ")
        print("#   (points_per_row,points_per_column).")
        print("#   for example, (7, 7) for a standard chessboard (not (8,8))")
        print("# You can enter (7,7) for a standard chessboard.")
        print("# For calibChessboardStereo() example, you can enter:")
        print("#   (7, 12)")
        patternSize = eval(input2())
        if (type(patternSize) != tuple or len(patternSize) != 2):
            print("# Error: patternSize needs to be a 2-integer tuple.")
            return
        print("# Number of inner corners is ", patternSize)
    # cellSize
    if (type(cellSize) == type(None)):
        print("# The calibChessboardStereo() needs cell size of a cell of the"
              " chessboard.")
        print("#   It is the physical size of a cell of chessboard. ")
        print("# For example, (50.8, 50.8)")
        print("# The unit can be mm, inch, m, or any length unit you prefer")
        print("#   but needs to be consistent through you entire analysis.")
        print("# For calibChessboardStereo() example, you can enter:")
        print("#   (21.31, 21.35)")
        cellSize = eval(input2())
        if (type(cellSize) != tuple or len(cellSize) != 2):
            print("# Error: cellSize needs to be a 2-integer tuple.")
            return
        print("# The cell size is ", cellSize)
    # imgSizeL and imgSizeR,
    # get image sizes by reading the first images in fileLists
    img = cv.imread(fileListL[0])
    if (type(img) != type(None) and img.shape[0] > 0):
        imgSizeL = (img.shape[1], img.shape[0])
    else:
        print("# Error: calibChessboardStereo(): Cannot find file: ",
              fileListL[0])
        zeroArray = np.zeros((0))
        return np.nan, zeroArray, zeroArray, zeroArray, zeroArray, zeroArray,\
               zeroArray
    img = cv.imread(fileListR[0])
    if (type(img) != type(None) and img.shape[0] > 0):
        imgSizeR = (img.shape[1], img.shape[0])
    else:
        print("# Error: calibChessboardStereo(): Cannot find file: ",
              fileListR[0])
        zeroArray = np.zeros((0))
        return np.nan, zeroArray, zeroArray, zeroArray, zeroArray, zeroArray,\
               zeroArray
    # initial guesses of cmat and dvec
    guessFxL = max(imgSizeL)
    guessFyL = guessFxL
    guessCxL = imgSizeL[0] * .5 - .5
    guessCyL = imgSizeL[1] * .5 - .5
    guessCmatL = np.array([[guessFxL, 0, guessCxL],
                          [0, guessFyL, guessCyL],[0,0,1.]])
    cmatL = guessCmatL
    guessFxR = max(imgSizeR)
    guessFyR = guessFxR
    guessCxR = imgSizeR[0] * .5 - .5
    guessCyR = imgSizeR[1] * .5 - .5
    guessCmatR = np.array([[guessFxR, 0, guessCxR],
                          [0, guessFyR, guessCyR],[0,0,1.]])
    cmatR = guessCmatR
    dvecL = np.zeros((1,14))
    dvecR = np.zeros((1,14))
    # flags
    if (type(flagsL) == type(None) or type(flagsR) == type(None)):
        print("# Enter calibration flags for LEFT and RIGHT cameras.")
        print("# Typical values are:")
        print("#     0: OpenCV default (calibrates fx, fy, cx, cy, k1, k2, "\
              "p1, p2, k3)")
        print("#     16385: (calibrates fx, fy, cx, cy, k1, k2, p1, p2, k3, "\
              "k4, k5, k6)")
        print("#     24705: (calibrates fx, fy, cx, cy, k1, k2, p1, p2,   , "\
              "k4, k5)")
        print("#     129: (calibrates fx, fy, cx, cy, k1, k2, p1, p2)")
        print("#     193: (calibrates fx, fy, cx, cy, k1, p1, p2)")
        print("#     201: (calibrates fx, fy, cx, cy, k1)")
        print("#     205: (calibrates fx, fy, k1)")
        print("#     207: (calibrates fx, k1)")
        print("# For example (two integers separated with a comma): ")
        print("#     201, 201")
        uInput = eval(input2())
        flagsL = uInput[0]
        flagsR = uInput[1]
        print("# The calibration flags are %d (left) and %d (right)" % \
              (flagsL, flagsR))
    # output directory
    if (type(outputDir) == type(None)):
        print("# Enter output directory of the chessboard stereo calibration.")
        print("# For example:")
        print("#   examples/calibChessboardStereo/cam5_cam6_stereoCalib/")
        outputDir = input2()
    outputDir = outputDir.strip() 
    if outputDir[-1] != '/' and outputDir[-1] != '\\':
        if outputDir.find('\\') >= 0:
            outputDir += "\\"
        else:
            outputDir += "/"
    print("# The output directory is ", outputDir)
    # allocate main arrays
    nImgPairs = len(fileListL)
    imgPoints2fL = np.zeros((nImgPairs, patternSize[0] * patternSize[1], 2),
                           dtype=np.float32)
    prjPoints2fL = np.zeros(imgPoints2fL.shape, dtype=imgPoints2fL.dtype)
    imgPoints2fR = np.zeros(imgPoints2fL.shape, dtype=imgPoints2fL.dtype)
    prjPoints2fR = np.zeros(imgPoints2fL.shape, dtype=imgPoints2fL.dtype)
    prjPoints2fR_stereo = np.zeros(
        (nImgPairs, patternSize[0] * patternSize[1], 2), dtype=np.float32)
    objPoints3f = np.zeros((nImgPairs, patternSize[0] * patternSize[1],3),
                            dtype=np.float32)
    idxOfFound = np.zeros((nImgPairs), dtype=np.bool)
    # find corners
    iFound = 0
    for i in range(nImgPairs):
        # find corners in LEFT calibration image
        filename = fileListL[i]
        img = cv.imread(filename, cv.IMREAD_GRAYSCALE)
        if (type(img) != type(None) and img.shape[0] > 0
            and img.shape[1] > 0):
            imgSizeL = (img.shape[1], img.shape[0])
        else:
            print("# Warning: calibChessboardStereo() cannot open file", \
                  filename)
        foundL, ptsThisL = cv.findChessboardCorners(img, patternSize)
        if foundL==False:
            print("# Warning: calibChessboardStereo() cannot find corners in"\
                  " left calibration %d: " % (i + 1), filename)
            continue
        # find corners in RIGHT calibration image
        filename = fileListR[i]
        img = cv.imread(filename, cv.IMREAD_GRAYSCALE)
        if (type(img) != type(None) and img.shape[0] > 0
            and img.shape[1] > 0):
            imgSizeR = (img.shape[1], img.shape[0])
        else:
            print("# Warning: calibChessboardStereo() cannot open file", \
                  filename)
        foundR, ptsThisR = cv.findChessboardCorners(img, patternSize)
        if foundR==False:
            print("# Warning: calibChessboardStereo() cannot find corners in"\
                  " right calibration %d: " % (i + 1), filename)
            continue
        # Corners of both images are found. Save the image points.
        # put left and right corners to imgPoints2fL and imgPoints2fR
        idxOfFound[i] = True
        imgPoints2fL[iFound,:,:] = ptsThisL[:,0,:]
        imgPoints2fR[iFound,:,:] = ptsThisR[:,0,:]
        print("# Calibration image %d corners detection: all found."
               % (i + 1))
        # object points
        for iy in range(patternSize[1]):
            for ix in range(patternSize[0]):
                objPoints3f[iFound, ix + iy * patternSize[0], 0] \
                    = ix * cellSize[0]
                objPoints3f[iFound, ix + iy * patternSize[0], 1] \
                    = iy * cellSize[1]
                objPoints3f[iFound, ix + iy * patternSize[0], 2] = 0.0
        iFound += 1
    nFound = iFound
    # do calibration
    retValL, cmatL, dvecL, rvecsL, tvecsL = cv.calibrateCamera(
        objPoints3f, imgPoints2fL, imageSize=imgSizeL,
        cameraMatrix=cmatL, distCoeffs=dvecL, flags=flagsL)
    retValR, cmatR, dvecR, rvecsR, tvecsR = cv.calibrateCamera(
        objPoints3f, imgPoints2fR, imageSize=imgSizeR,
        cameraMatrix=cmatR, distCoeffs=dvecR, flags=flagsR)
#    stereoCalibFlags = 0
#    stereoCalibFlags = cv.CALIB_FIX_INTRINSIC  # OpenCV default value is cv2.CALIB_FIX_INTRINSIC
    retVal, cmatL, dvecL, cmatR, dvecR, R, T, E, F = cv.stereoCalibrate(
        objPoints3f, imgPoints2fL, imgPoints2fR, cmatL, dvecL, cmatR, dvecR,
        imgSizeL)
#        flags=stereoCalibFlags)
    # R, T --> r44RL, rvec, tvec
    r44RL = np.eye(4, dtype=np.float64)
    r44RL[0:3,0:3] = R
    r44RL[0:3,3] = T.flatten()
    rvecL = np.zeros((3,1), np.float64)
    tvecL = np.zeros((3,1), np.float64)
    rvecR, jacob = cv.Rodrigues(R)
    tvecR = T.reshape((3,1))
    # calculate projected points
    objPoints = np.array(objPoints3f[0,:,:], dtype=float)
    iFound = 0
    for i in range(nImgPairs):
        if idxOfFound[i] == True:
            # project left points
            prjPoints, jacobian = cv.projectPoints(
                objPoints, rvecsL[iFound], tvecsL[iFound], cmatL, dvecL)
            prjPoints = prjPoints.reshape(-1,2)
            prjPoints2fL[iFound,:,:] = prjPoints[:,:]
            # project right points
            prjPoints, jacobian = cv.projectPoints(
                objPoints, rvecsR[iFound], tvecsR[iFound], cmatR, dvecR)
            prjPoints = prjPoints.reshape(-1,2)
            prjPoints2fR[iFound,:,:] = prjPoints[:,:]
            # project right points based on rvecsL and calibrated R and T
            r44L_thisBoard = np.eye(4, dtype=np.float64)
            r44L_thisBoard[0:3,0:3] = cv.Rodrigues(rvecsL[iFound])[0]
            r44L_thisBoard[0:3,3] = tvecsL[iFound].flatten()
#            rvecR_thisBoard = np.matmul(r44L_thisBoard,
#                                        np.linalg.inv(r44RL))
            rvecR_thisBoard = np.matmul(r44RL, r44L_thisBoard)
            rvec_thisBoard = cv.Rodrigues(rvecR_thisBoard[0:3,0:3])[0]
            tvec_thisBoard = rvecR_thisBoard[0:3,3]
            prjPoints, jacobian = cv.projectPoints(
                objPoints, rvec_thisBoard, tvec_thisBoard, cmatR, dvecR)
            prjPoints = prjPoints.reshape(-1,2)
            prjPoints2fR_stereo[iFound,:,:] = prjPoints[:,:]
            #
            iFound += 1
    # write camera parameters
    writeCamera(outputDir + "camera_left.csv", rvecL, tvecL, cmatL, dvecL)
    writeCamera(outputDir + "camera_right.csv", rvecR, tvecR, cmatR, dvecR)
    iFound = 0
    for i in range(nImgPairs):
        if idxOfFound[i] == True:
            writeCamera(outputDir + "camera_left_wrt_Chessboard_%03d.csv" % \
                        (i + 1), rvecsL[iFound], tvecsL[iFound], cmatL, dvecL)
            writeCamera(outputDir + "camera_right_wrt_Chessboard_%03d.csv" % \
                        (i + 1), rvecsR[iFound], tvecsR[iFound], cmatR, dvecR)
    np.savetxt(outputDir + "camera_R33.csv", R, fmt='%24.16e',
               delimiter=' , ', header='Stereo calibration R matrix')
    np.savetxt(outputDir + "camera_T31.csv", T, fmt='%24.16e',
               delimiter=' , ', header='Stereo calibration T vector')
    np.savetxt(outputDir + "camera_E33.csv", E, fmt='%24.16e',
               delimiter=' , ', header='Stereo calibration E matrix')
    np.savetxt(outputDir + "camera_F33.csv", F, fmt='%24.16e',
               delimiter=' , ', header='Stereo calibration F matrix')
    # write image (corner) points files
    iFound = 0
    for i in range(nImgPairs):
        if idxOfFound[i] == True:
            writePoints(outputDir + "imgPoints_Left_%03d.csv" % (i + 1),
                        imgPoints2fL[iFound,:,:])
            writePoints(outputDir + "imgPoints_Right_%03d.csv" % (i + 1),
                        imgPoints2fR[iFound,:,:])
            iFound += 1
    # write object points files
    writePoints(outputDir + "objPoints.csv", objPoints3f[0,:,:])
    # write projection points to file
    iFound = 0
    for i in range(nImgPairs):
        if idxOfFound[i] == True:
            writePoints(outputDir + "prjPoints_Left_%03d.csv" % (i + 1),
                        prjPoints2fL[iFound,:,:])
            writePoints(outputDir + "prjPoints_Right_%03d.csv" % (i + 1),
                        prjPoints2fR[iFound,:,:])
            iFound += 1
    # write projection errors to file
    iFound = 0
    for i in range(nImgPairs):
        if idxOfFound[i] == True:
            writePoints(outputDir + "prjErrors_Left_%03d.csv" % (i + 1),
                        prjPoints2fL[iFound,:,:] - imgPoints2fL[iFound,:,:])
            writePoints(outputDir + "prjErrors_Right_%03d.csv" % (i + 1),
                        prjPoints2fR[iFound,:,:] - imgPoints2fR[iFound,:,:])
            iFound += 1
    # draw calibration points and projected points on the image
    iFound = 0
    for i in range(nImgPairs):
        if idxOfFound[i] == True:
            # Left
            saveImgFile_i = outputDir + "prjImg_Left_%03d" % (i + 1) + ".JPG"
            img = cv.imread(fileListL[i])
            img = drawPoints(img, imgPoints2fL[iFound,:,:], color=[0,0,0],
                             markerSize=20, thickness=5, savefile=".")
            img = drawPoints(img, imgPoints2fL[iFound,:,:], color=[255,0,0],
                             markerSize=18, thickness=3, savefile=".")
            img = drawPoints(img, prjPoints2fL[iFound,:,:], color=[0,0,0],
                             markerSize=20, thickness=5, savefile=".")
            img = drawPoints(img, prjPoints2fL[iFound,:,:], color=[0,0,255],
                             markerSize=18, thickness=3,
                             savefile=saveImgFile_i)
            # Right (based on rvec)
            saveImgFile_i = outputDir + "prjImg_Right_%03d" % (i + 1) + ".JPG"
            img = cv.imread(fileListR[i])
            img = drawPoints(img, imgPoints2fR[iFound,:,:], color=[0,0,0],
                             markerSize=20, thickness=5, savefile=".")
            img = drawPoints(img, imgPoints2fR[iFound,:,:], color=[255,0,0],
                             markerSize=18, thickness=3, savefile=".")
            img = drawPoints(img, prjPoints2fR[iFound,:,:], color=[0,0,0],
                             markerSize=20, thickness=5, savefile=".")
            img = drawPoints(img, prjPoints2fR[iFound,:,:], color=[0,0,255],
                             markerSize=18, thickness=3,
                             savefile=saveImgFile_i)
            # Right based on based on rvecsL and calibrated R and T
            saveImgFile_i = outputDir + "prjImg_basedOn_stereo_%03d" % (i + 1) + ".JPG"
            img = cv.imread(fileListR[i])
            img = drawPoints(img, imgPoints2fR[iFound,:,:], color=[0,0,0],
                            markerSize=20, thickness=5, savefile=".")
            img = drawPoints(img, imgPoints2fR[iFound,:,:], color=[255,0,0],
                            markerSize=18, thickness=3, savefile=".")
            img = drawPoints(img, prjPoints2fR_stereo[iFound,:,:], color=[0,0,0],
                            markerSize=20, thickness=5, savefile=".")
            img = drawPoints(img, prjPoints2fR_stereo[iFound,:,:], color=[0,0,255],
                            markerSize=18, thickness=3,
                            savefile=saveImgFile_i)
            iFound += 1
    # final information
    print("# The output results are stored in directory: ", outputDir)
    print("# camera_left.csv and camera_right.csv store the calibrated "
          "parameters of the left and right cameras")
    print("# imgPoints....csv store the image points of chessboard corners.")
    print("# prjPoints....csv store the projected points.")
    print("# prjErrors....csv store the projection errors.")
    print("# prjImg....JPG show the chessboard corners (blue) and projection"
          " points (red) in images.")
    #
    return retVal, cmatL, dvecL, cmatR, dvecR, R, T, E, F


def calibStereo_test():
    fileListL = createFileList("examples/calibChessboardStereo/cam5_calib/"
                                "G*.JPG", ".")
    fileListR = createFileList("examples/calibChessboardStereo/cam6_calib/"
                                "G*.JPG", ".")
    patternSize = (12,7)
    cellSize = (21.31, 21.35)
    flagsL = 201 # only calibrates fx, fy, cx, cy, and k1.
    flagsR = 201 # only calibrates fx, fy, cx, cy, and k1.
        # 0: OpenCV default (calibrates fx, fy, cx, cy, k1, k2, p1, p2, k3)
        # 16385: (calibrates fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6)
        # 24705: (calibrates fx, fy, cx, cy, k1, k2, p1, p2,   , k4, k5)
        # 129: (calibrates fx, fy, cx, cy, k1, k2, p1, p2)
        # 193: (calibrates fx, fy, cx, cy, k1, p1, p2)
        # 201: (calibrates fx, fy, cx, cy, k1)
        # 205: (calibrates fx, fy, k1)
        # 207: (calibrates fx, k1)
    outputDir = "examples/calibChessboardStereo/cam5_cam6_stereoCalib/"
    #
    calibChessboardStereo(fileListL=fileListL,
                              fileListR=fileListR,
                              patternSize=patternSize,
                              cellSize=cellSize,
                              flagsL=flagsL,
                              flagsR=flagsR,
                              outputDir=outputDir)


def calibStereo_test_brb_example():
    # Left
    path_left = "D:/ExpDataSamples/20220500_Brb/brb1/" \
               "brb1_cam5_neck_calib/"
    path_left_out = path_left + "calib_FxFyCxCyK1/"
    fList_left = createFileList(path_left + "G00*.JPG", ".") # , or
    pSize_left = ((7, 12))
    cSize_left = (21.31, 21.35)
    flags_left = 201 # only calibrates fx, fy, cx, cy, and k1.
        # 0: OpenCV default (calibrates fx, fy, cx, cy, k1, k2, p1, p2, k3)
        # 16385: (calibrates fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6)
        # 24705: (calibrates fx, fy, cx, cy, k1, k2, p1, p2,   , k4, k5)
        # 129: (calibrates fx, fy, cx, cy, k1, k2, p1, p2)
        # 193: (calibrates fx, fy, cx, cy, k1, p1, p2)
        # 201: (calibrates fx, fy, cx, cy, k1)
        # 205: (calibrates fx, fy, k1)
        # 207: (calibrates fx, k1)
    # criteria = (cv.TERM_CRITERIA_EPS + cv.TermCriteria_COUNT, 40, 0.001)
    camFile_left = path_left_out + "brb1_cam5_parameters.csv"
    imgPointsFile_left = path_left_out + "brb1_cam5_imgPoints.csv"
    objPointsFile_left = path_left_out + "brb1_cam5_objPoints.csv"
    prjPointsFile_left = path_left_out + "brb1_cam5_proj_test.csv"
    prjErrorsFile_left = path_left_out + "brb1_cam5_err_test.csv"
    prjImagesFile_left = path_left_out + "brb1_cam5_proj_img.JPG"
    # run calibChessboard, optionally not giving cmat, dvec, and criteria
    retVal, cmat, dvec, rvec, tvec = calibChessboard(
        fileList          = fList_left,
        patternSize       = pSize_left,
        cellSize          = cSize_left,
        flags             = flags_left,
        saveCamFile       = camFile_left,
        saveImgPointsFile = imgPointsFile_left,
        saveObjPointsFile = objPointsFile_left,
        savePrjFile       = prjPointsFile_left,
        saveErrFile       = prjErrorsFile_left,
        saveImgFile       = prjImagesFile_left)
    # Right
    path_right = "D:/ExpDataSamples/20220500_Brb/brb1/" \
               "brb1_cam5_neck_calib/"
    path_right_out = path_right + "calib_FxFyCxCyK1/"
    fList_right = createFileList(path_right + "G00*.JPG", ".") # , or
    pSize_right = (7, 12)
    cSize_right = (21.31, 21.35)
    flags_right = 201 # only calibrates fx, fy, cx, cy, and k1.
        # 0: OpenCV default (calibrates fx, fy, cx, cy, k1, k2, p1, p2, k3)
        # 16385: (calibrates fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6)
        # 24705: (calibrates fx, fy, cx, cy, k1, k2, p1, p2,   , k4, k5)
        # 129: (calibrates fx, fy, cx, cy, k1, k2, p1, p2)
        # 193: (calibrates fx, fy, cx, cy, k1, p1, p2)
        # 201: (calibrates fx, fy, cx, cy, k1)
        # 205: (calibrates fx, fy, k1)
        # 207: (calibrates fx, k1)
    # criteria = (cv.TERM_CRITERIA_EPS + cv.TermCriteria_COUNT, 40, 0.001)
    camFile_right = path_right_out + "brb1_cam5_parameters.csv"
    imgPointsFile_right = path_right_out + "brb1_cam5_imgPoints.csv"
    objPointsFile_right = path_right_out + "brb1_cam5_objPoints.csv"
    prjPointsFile_right = path_right_out + "brb1_cam5_proj_test.csv"
    prjErrorsFile_right = path_right_out + "brb1_cam5_err_test.csv"
    prjImagesFile_right = path_right_out + "brb1_cam5_proj_img.JPG"
    # run calibChessboard, optionally not giving cmat, dvec, and criteria
    retVal, cmat, dvec, rvec, tvec = calibChessboard(
        fileList          = fList_right,
        patternSize       = pSize_right,
        cellSize          = cSize_right,
        flags             = flags_right,
        saveCamFile       = camFile_right,
        saveImgPointsFile = imgPointsFile_right,
        saveObjPointsFile = objPointsFile_right,
        savePrjFile       = prjPointsFile_right,
        saveErrFile       = prjErrorsFile_right,
        saveImgFile       = prjImagesFile_right)
