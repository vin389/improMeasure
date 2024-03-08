import numpy as np
import cv2 as cv

from permuteRows import permuteRows
from validsOfPoints import validsOfPoints
# from validsOfPoints2 import validsOfPoints2
from projectPoints import projectPoints


def calibrateCamera(objPoints, imgPoints, imgSize, cmat, dvec, 
                    rvecs, tvecs, flags):
    """
    Before calling cv.calibrateCamera, this function removes nan points. 
    In addition to returning ret (error), cmat, dvec, rvec, and tvec, this 
    function returns prjPoints (projected points in image) and prjErrors
    (prjPoints - imgPoints) which may contain np.nan if imgPoints contains
    np.nan.
    See demonstration code in test_calibrateCamera()

    Parameters
    ----------
    objPoints : TYPE
        DESCRIPTION.
    imgPoints : TYPE
        DESCRIPTION.
    imgSize : TYPE
        DESCRIPTION.
    cmat : TYPE
        DESCRIPTION.
    dvec : TYPE
        DESCRIPTION.
    rvecs : TYPE
        DESCRIPTION.
    tvecs : TYPE
        DESCRIPTION.
    flags : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    # confirm # of images is one. This function only accepts single-image 
    # calibration.
    if len(objPoints.shape) == 3 and objPoints.shape[0] != 1:
        print("# Error: This calibrateCamera() does not support multi-image calibration. Use cv.calibrateCamera instead.")
        return
    # reshape object points and image points
    objPoints_reshaped = objPoints.reshape((-1, 3))
    imgPoints_reshaped = imgPoints.reshape((-1, 2))
    # create points3d_valid that contains valid points of points3d
    # (assuming some points in points3d are nan points)
    nPointsAll = objPoints_reshaped.shape[0]
    if imgPoints.shape[0] != nPointsAll:
        print("# Error: calibrateCamera: number of image points must be the same as number of object points.")
        return
    # calculate valids (size: nPoints), nPointsValid, 
    # idx_o2n (size: nPointsAll), and idx_n2o(size: nPointsValid)
    valids_objPoints, idx_o2n, idx_n2o = validsOfPoints(objPoints_reshaped)
    valids_imgPoints, idx_o2n, idx_n2o = validsOfPoints(imgPoints_reshaped)
    valids = valids_objPoints
    for i in range(nPointsAll):
        if valids_imgPoints[i] == 0:
            valids[i] = 0
    # calculate nPointsValid (number of valid points) and idx_o2n
    nPointsValid = 0
    idx_o2n[:] = -1
    for i in range(nPointsAll):
        if valids[i] != 0:
            idx_o2n[i] = nPointsValid
            nPointsValid += 1
    # indices mapping all points (0:nPointsAll) to valid points
    idx_n2o = np.ones((nPointsValid), dtype=int) * (-1)
    for i in range(nPointsAll):
        if idx_o2n[i] >= 0:
            idx_n2o[idx_o2n[i]] = i
    # valid points
    objPoints_valid = permuteRows(objPoints_reshaped, idx_n2o).reshape((1,-1,3)).astype(np.float32)
    imgPoints_valid = permuteRows(imgPoints_reshaped, idx_n2o).reshape((1,-1,2)).astype(np.float32)
    # calibrate camera
    ret, cmat, dvec, rvecs, tvecs = cv.calibrateCamera(
        objPoints_valid, imgPoints_valid, imgSize, cmat, dvec, rvecs, tvecs, flags)
#    points3d_valid = points3d[idx_n2o]
    # project only the valid points
#    imgPoints_valid, jacob_valid = cv.projectPoints(
#        points3d_valid, rvec, tvec, cmat, dvec)
    # project points
    prjPoints, jacob = projectPoints(objPoints, rvecs[0], tvecs[0], 
                                     cmat, dvec)
    prjErrors = prjPoints.reshape((-1, 2)) - imgPoints_reshaped
    # return
    return ret, cmat, dvec, rvecs[0], tvecs[0], prjPoints, prjErrors


def test_calibrateCamera():
    objPoints = np.array([[50.0,0.0,50.0],[190.0,0.0,50.0],[330.0,0.0,50.0],[500.0,0.0,50.0],[670.0,0.0,50.0],[830.0,0.0,50.0],[1000.0,0.0,50.0],[1170.0,0.0,50.0],[1310.0,0.0,50.0],[1450.0,0.0,50.0],[50.0,0.0,190.0],[190.0,0.0,190.0],[330.0,0.0,190.0],[500.0,0.0,190.0],[670.0,0.0,190.0],[830.0,0.0,190.0],[1000.0,0.0,190.0],[1170.0,0.0,190.0],[1310.0,0.0,190.0],[1450.0,0.0,190.0],[50.0,0.0,350.0],[190.0,0.0,350.0],[330.0,0.0,350.0],[500.0,0.0,350.0],[670.0,0.0,350.0],[830.0,0.0,350.0],[1000.0,0.0,350.0],[1170.0,0.0,350.0],[1310.0,0.0,350.0],[1450.0,0.0,350.0],[50.0,0.0,510.0],[190.0,0.0,510.0],[330.0,0.0,510.0],[500.0,0.0,510.0],[670.0,0.0,510.0],[830.0,0.0,510.0],[1000.0,0.0,510.0],[1170.0,0.0,510.0],[1310.0,0.0,510.0],[1450.0,0.0,510.0],[50.0,0.0,670.0],[190.0,0.0,670.0],[330.0,0.0,670.0],[500.0,0.0,670.0],[670.0,0.0,670.0],[830.0,0.0,670.0],[1000.0,0.0,670.0],[1170.0,0.0,670.0],[1310.0,0.0,670.0],[1450.0,0.0,670.0],[50.0,0.0,830.0],[190.0,0.0,830.0],[330.0,0.0,830.0],[500.0,0.0,830.0],[670.0,0.0,830.0],[830.0,0.0,830.0],[1000.0,0.0,830.0],[1170.0,0.0,830.0],[1310.0,0.0,830.0],[1450.0,0.0,830.0],[50.0,0.0,990.0],[190.0,0.0,990.0],[330.0,0.0,990.0],[500.0,0.0,990.0],[670.0,0.0,990.0],[830.0,0.0,990.0],[1000.0,0.0,990.0],[1170.0,0.0,990.0],[1310.0,0.0,990.0],[1450.0,0.0,990.0],[50.0,0.0,1150.0],[190.0,0.0,1150.0],[330.0,0.0,1150.0],[500.0,0.0,1150.0],[670.0,0.0,1150.0],[830.0,0.0,1150.0],[1000.0,0.0,1150.0],[1170.0,0.0,1150.0],[1310.0,0.0,1150.0],[1450.0,0.0,1150.0],[50.0,0.0,1310.0],[190.0,0.0,1310.0],[330.0,0.0,1310.0],[500.0,0.0,1310.0],[670.0,0.0,1310.0],[830.0,0.0,1310.0],[1000.0,0.0,1310.0],[1170.0,0.0,1310.0],[1310.0,0.0,1310.0],[1450.0,0.0,1310.0],[50.0,0.0,1450.0],[190.0,0.0,1450.0],[330.0,0.0,1450.0],[500.0,0.0,1450.0],[670.0,0.0,1450.0],[830.0,0.0,1450.0],[1000.0,0.0,1450.0],[1170.0,0.0,1450.0],[1310.0,0.0,1450.0],[1450.0,0.0,1450.0]],dtype=float)
    imgPoints = np.array([[2169.6, 2499.2], [2350.2, 2497.3], [2517.3, 2501.1], [2733.3, 2503.5], [np.nan, np.nan], [3138.7, 2506.0], [3342.6, 2506.9], [np.nan, np.nan], [3719.4, 2510.5], [np.nan, np.nan], [2171.9, 2320.1], [2349.8, 2322.0], [2520.6, 2324.1], [2737.6, 2327.4], [np.nan, np.nan], [3140.6, 2333.5], [3343.3, 2334.1], [np.nan, np.nan], [3716.3, 2338.7], [np.nan, np.nan], [2175.4, 2123.8], [2354.0, 2126.7], [2524.5, 2128.7], [2737.9, 2131.3], [2944.0, 2135.8], [3138.3, 2137.6], [3342.8, 2140.0], [3547.5, 2142.4], [3714.5, 2146.1], [3881.2, 2148.4], [2179.1, 1922.4], [2356.9, 1925.7], [2527.9, 1928.7], [2740.3, 1934.9], [2944.9, 1936.3], [3137.7, 1940.5], [3342.0, 1944.6], [3544.4, 1948.3], [3712.3, 1950.9], [3877.8, 1953.3], [2180.2, 1728.4], [2359.3, 1731.4], [2527.9, 1734.6], [2740.5, 1739.1], [2945.5, 1743.3], [3137.3, 1747.5], [3341.9, 1750.4], [3543.1, 1755.2], [3709.6, 1758.3], [3874.5, 1761.5], [2185.1, 1533.1], [2360.4, 1537.6], [2532.9, 1541.7], [2741.6, 1546.2], [2946.0, 1551.5], [3136.3, 1555.6], [3341.1, 1562.5], [3542.4, 1565.7], [3706.6, 1568.0], [3872.0, 1572.9], [2188.3, 1336.0], [2363.5, 1343.6], [2533.3, 1346.1], [2741.3, 1352.4], [2945.7, 1358.7], [3137.2, 1363.7], [3339.3, 1369.5], [3539.4, 1375.5], [3703.1, 1380.1], [3866.9, 1382.8], [2192.3, 1145.5], [2366.3, 1151.4], [2535.5, 1154.9], [2741.8, 1163.1], [2944.9, 1169.4], [3136.4, 1173.9], [3337.5, 1180.3], [3537.2, 1187.1], [3703.4, 1191.8], [3865.1, 1196.5], [2196.0, 952.5], [2367.1, 959.3], [2536.6, 963.3], [2742.8, 972.5], [2945.0, 978.7], [3137.0, 986.6], [3334.2, 991.4], [3536.6, 996.7], [3700.4, 1004.1], [3865.1, 1010.2], [2199.5, 795.2], [2369.0, 800.0], [2539.2, 808.3], [2741.0, 814.4], [2946.4, 819.7], [3135.8, 826.0], [3335.3, 832.4], [3534.8, 834.0], [3699.5, 839.3], [3861.6, 845.1]], dtype=float)
    imgSize = (6000, 3375)
    cmat = np.array([[5000.,0,2999.5],[0,5000.,1687.],[0,0,1.]])
    dvec = np.array([0.,0,0,0])
    rvecs = np.zeros(3, dtype=float)
    tvecs = np.zeros(3, dtype=float)
    flags = 14541 # fx,fy,k1,,cx=(w-1)/2,cy=(h-1)/2
    ret, cmat, dvec, rvec, tvec, prjPoints, prjErrors = calibrateCamera(
        objPoints, imgPoints, imgSize, cmat, dvec, rvecs, tvecs, flags)
    # ret would be 2.37... (the error returned from OpenCV calibrateCamera)
    # cmat: a 3x3 float64 np.ndarray
    # dvec: a 5x1 (or longer) float64 np.ndarray
    # rvec: a 3x1 float64 np.ndarray (not a tuple like OpenCV returns as 
    #       this function only handles single image)
    # tvec: a 3x1 float64 np.ndarray
    # prjPoints: an Nx2 float64 np.ndarray, N is number of points 
    # prjErrors: an Nx2 float64 np.ndarray. It can contain nan.
    
    
    
    
    
    
    
    
    
    
    