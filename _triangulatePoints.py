import cv2 as cv
import numpy as np 

def triangulatePoints2(
        cmat1, dvec1, rvec1, tvec1, 
        cmat2, dvec2, rvec2, tvec2, 
        imgPoints1, imgPoints2):
    """
    This function triangulates points from given two sets of image 
    coordinates of N points, intrinsic and extrinsic parameters of 
    two cameras.   
    
    Parameters
    ----------
    cmat1 : numpy ndarray, a 3x3 np.float64 numpy matrix
    dvec1 : numpy ndarray, a 1D np.floatr64 numpy matrix
    rvec1 : numpy ndarray, 3-element np.float64 numpy matrix
    tvec1 : numpy ndarray, 3-element np.float64 numpy matrix
    cmat2 : numpy ndarray, a 3x3 np.float64 numpy matrix
    dvec2 : numpy ndarray, a 1D np.floatr64 numpy matrix
    rvec2 : numpy ndarray, 3-element np.float64 numpy matrix
    tvec2 : numpy ndarray, 3-element np.float64 numpy matrix
    imgPoints1 : numpy ndarray, Nx2 2D array of N points 
        image coordinates of N points in camera 1 (in original photo)
    imgPoints2 : numpy ndarray, Nx2 2D array of N points 
        image coordinates of N points in camera 2 (in original photo)

    Returns
    -------
    objPoints : numpy ndarray, Nx3 np.float64 numpy matrix
        object points triangulated, in world coordinate
    objPoints1 : numpy ndarray, Nx3 np.float64 numpy matrix
        object points triangulated, in camera-1 coordinate
    objPoints2 : numpy ndarray, Nx3 np.float64 numpy matrix
        object points triangulated, in camera-2 coordinate
    prjPoints1 : numpy ndarray, Nx2 np.float64 numpy matrix
        projected points in camera-1 image coordinate
    prjPoints2 : numpy ndarray, Nx2 np.float64 numpy matrix
        projected points in camera-2 image coordinate
    prjErrors1 : numpy ndarray, Nx2 np.float64 numpy matrix
        projected errors in camera-1 image coordinate
        i.e., prjPoints1 - imgPoints1
    prjErrors2 : numpy ndarray, Nx2 np.float64 numpy matrix
        projected errors in camera-2 image coordinate
        i.e., prjPoints2 - imgPoints2


    """
    # force reshape and type conversion 
    imgPoints1 = np.array(imgPoints1, dtype=np.float64).reshape(-1,2)
    imgPoints2 = np.array(imgPoints2, dtype=np.float64).reshape(-1,2)
    # initialization (to zero-size arrays)
    objPoints = np.zeros((0))
    prjPoints1 = np.zeros((0))
    prjPoints2 = np.zeros((0))
    prjErrors1 = np.zeros((0))
    prjErrors2 = np.zeros((0))
    # check 
    nPoints = imgPoints1.shape[0]
    if imgPoints2.shape[0] != nPoints:
        print("# Error: triangulatePoints2(): imgPoints1 and 2 have"
              " different number of points.")
        return objPoints, prjPoints1, prjPoints2, prjErrors1, prjErrors2 
    # memory allocation
    objPoints = np.ones((nPoints, 3), dtype=np.float64) * np.nan
#    prjPoints1 = np.ones((nPoints, 2), dtype=np.float64) * np.nan
#    prjPoints2 = np.ones((nPoints, 2), dtype=np.float64) * np.nan
#    prjErrors1 = np.ones((nPoints, 2), dtype=np.float64) * np.nan
#    prjErrors2 = np.ones((nPoints, 2), dtype=np.float64) * np.nan
    prjMat1 = np.zeros((3,4), dtype=np.float64)
    prjMat2 = np.zeros((3,4), dtype=np.float64)
    rctMat1 = np.zeros((3,3), dtype=np.float64)
    rctMat2 = np.zeros((3,3), dtype=np.float64)
    qMat = np.zeros((3,4), dtype=np.float64)
    undPoints1 = np.ones((nPoints, 2), dtype=np.float64) * np.nan
    undPoints2 = np.ones((nPoints, 2), dtype=np.float64) * np.nan
    # Calculate rmat, tvec from coord of left to right 
    r44L = np.eye(4, dtype=np.float64)
    r44R = np.eye(4, dtype=np.float64)
    r44L[0:3,0:3] = cv.Rodrigues(rvec1)[0]
    r44L[0:3,3] = tvec1.flatten()
    r44R[0:3,0:3] = cv.Rodrigues(rvec2)[0]
    r44R[0:3,3] = tvec2.flatten()
    r44 = np.matmul(r44R, np.linalg.inv(r44L))
    r33 = r44[0:3,0:3].copy()
    rvec = cv.Rodrigues(r33)[0]
    tvec = r44[0:3,3].copy()
    # stereo rectify
    rctMat1, rctMat2, prjMat1, prjMat2, qMat, dum1, dum2 = \
        cv.stereoRectify(cmat1, dvec1, cmat2, dvec2, (1000,1200), r33, tvec)
    # undistortion
    #    cv::undistortPoints(xLC2, uL, camMatrix1, distVect1, rcL, pmL);
    #    cv::undistortPoints(xRC2, uR, camMatrix2, distVect2, rcR, pmR);
    undPoints1 = cv.undistortPoints(imgPoints1, cmat1, dvec1, undPoints1, 
                                    rctMat1, prjMat1).reshape(-1,2)
    undPoints2 = cv.undistortPoints(imgPoints2, cmat2, dvec2, undPoints2, 
                                    rctMat2, prjMat2).reshape(-1,2)
    # triangulation 
    objPoints = cv.triangulatePoints(prjMat1, prjMat2, undPoints1.transpose(), 
                                     undPoints2.transpose())
    # coordinate transformation to cam-1 coord.
    rctInv1 = np.eye(4, dtype=np.float64)
    rctInv1[0:3,0:3] = np.linalg.inv(rctMat1)
    objPoints = np.matmul(rctInv1, objPoints)
    # object points in cam1, cam2, world coordinate
    objPoints1 = objPoints.copy()
    objPoints2 = np.matmul(r44, objPoints1)
    objPoints = np.matmul(np.linalg.inv(r44L), objPoints).transpose()
    objPoints1 = objPoints1.transpose()
    objPoints2 = objPoints2.transpose()
    for iPt in range(objPoints.shape[0]):
        for ix in range(3):
            objPoints[iPt,ix] /= objPoints[iPt,3]
            objPoints1[iPt,ix] /= objPoints1[iPt,3]
            objPoints2[iPt,ix] /= objPoints2[iPt,3]
    objPoints = objPoints[:,0:3]
    objPoints1 = objPoints1[:,0:3]
    objPoints2 = objPoints2[:,0:3]
    # project points 
    prjPoints1 = cv.projectPoints(objPoints, rvec1, tvec1, cmat1, dvec1)[0].reshape(-1,2)
    prjPoints2 = cv.projectPoints(objPoints, rvec2, tvec2, cmat2, dvec2)[0].reshape(-1,2)
    # projection errors
    prjErrors1 = prjPoints1 - imgPoints1
    prjErrors2 = prjPoints2 - imgPoints2
    
    return objPoints, objPoints1, objPoints2, prjPoints1, prjPoints2, prjErrors1, prjErrors2
