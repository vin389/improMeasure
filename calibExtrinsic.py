import numpy as np
import cv2 as cv
from inputs import input2, input3
from imread2 import imread2
from readPoints import readPoints
from writePoints import writePoints
from readCamera import readCamera
from writeCamera import writeCamera 
from removeNanPoints import removeNanPoints
from drawPoints import drawPoints


def calibExtrinsic(img=None, objPoints=None, imgPoints=None, 
                   cmat=None, dvec=None,
                   saveCamFile=None, savePrjFile=None, saveErrFile=None, 
                   saveImgFile=None):
    """
    Given image points, object points, intrinsic parameters (cmat and dvec)
    and the image itself, this function
    estimates the extrinsic parameters of the camrea. 
    For example, 
        img = cv.imread('examples/calibExtrinsic/brb2_cam6.JPG')
        objPoints = readPoints('examples/calibExtrinsic/objPoints.csv')
        imgPoints = readPoints('examples/calibExtrinsic/imgPoints.csv')
        rvec,tvec,cmat,dvec = readCamera(
            'examples\\calibExtrinsic\\brb2_cam6.csv')
        retVal, cmat, dvec, rvec, tvec = calibExtrinsic(
            img=img, objPoints=objPoints, imgPoints=imgPoints,
            cmat=cmat, dvec=dvec, 
            saveCamFile = 'examples/calibExtrinsic/brb2_cam6_calibrated.csv', 
            saveImgFile = 'examples/calibExtrinsic/prj_brb2_cam6.jpg', 
            savePrjFile = 'examples/calibExtrinsic/prjPoints.csv',
            saveErrFile = 'examples/calibExtrinsic/errPoints.csv')
        
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
        If not given, this function asks the user.
    cmat : an 3-by-3 float numpy. 
        Initial guess of camera matrix. 
        If not given, this function asks the user.
    dvec : an 1 dimensional numpy , optional
        Initial guessed distortion coefficients.
        If not given, this function asks the user.
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
        The projected points are makred with green crosses.
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

    # camera matrix
    if type(cmat) == type(None) or type(dvec) == type(None):
        # get camera matrix from user
        rvec, tvec, cmat, dvec = readCamera()
            
    # remove nan points
    objPoints, imgPoints = removeNanPoints(objPoints, imgPoints)
    
    # convert to format for cv.solvePnP
    imgPoints2f = np.array(imgPoints, dtype=np.float32)
    imgPoints2f = imgPoints2f.reshape((1,-1,2))
    objPoints3f = np.array(objPoints, dtype=np.float32)
    objPoints3f = objPoints3f.reshape((1,-1,3))

    # call function cv.solvePnP()
    retVal, rvec, tvec = cv.solvePnP(
        objPoints3f, imgPoints2f, cmat, dvec)
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

def test_calibExtrinsic():
    img = cv.imread('examples/calibExtrinsic/brb2_cam6.JPG')
    objPoints = readPoints('examples/calibExtrinsic/objPoints.csv')
    imgPoints = readPoints('examples/calibExtrinsic/imgPoints.csv')
    rvec,tvec,cmat,dvec = readCamera('examples\\calibExtrinsic\\brb2_cam6.csv')
    retVal, cmat, dvec, rvec, tvec = calibExtrinsic(
        img=img, objPoints=objPoints, imgPoints=imgPoints,
        cmat=cmat, dvec=dvec, 
        saveCamFile = 'examples/calibExtrinsic/brb2_cam6_calibrated.csv', 
        saveImgFile = 'examples/calibExtrinsic/prj_brb2_cam6.jpg', 
        savePrjFile = 'examples/calibExtrinsic/prjPoints.csv',
        saveErrFile = 'examples/calibExtrinsic/errPoints.csv')

# test_calibExtrinsic()


                 


