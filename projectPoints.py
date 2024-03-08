import numpy as np
import cv2 as cv

from permuteRows import permuteRows
from validsOfPoints import validsOfPoints

def projectPoints(points3d: np.ndarray, rvec, tvec, cmat, dvec):
    """
    In addition to calling OpenCV cv.projectPoints(), this function allows
    some of the 3D points to be nan points. An nan point is a point that 
    contains nan in its coordinate. 
    For example, 
    rvec = np.array([1.4, -0.1, 0.1])
    tvec = np.array([-933, 8., 5511])
    cmat = np.array([[6800, 0, 3200],[0, 6800, 2500],[0, 0, 1.]])
    dvec = np.array([0.04, 0, 0, 0])
    points3d = np.array([[50, 0., 50],[np.nan, np.nan, np.nan],[100, 0, 50]])
    points2d, jacob = projectPoints(points3d, rvec, tvec, cmat, dvec)
    Then points2d would be
    array([[2111.0840601 , 2449.80610797],
           [          nan,           nan],
           [2173.47356039, 2450.54288129]])
    jacob would be
    array([[-5.60064230e+00,  4.21903508e+01,  3.73764424e+01,
             1.23436743e+00,  1.16128480e-04,  1.97463354e-01,
            -1.60134697e-01,  0.00000000e+00,  1.00000000e+00,
             0.00000000e+00, -2.78965970e+01, -7.15407307e-01,
             1.60426375e+01,  5.22417836e+02],
           [-1.48272440e+01,  3.65332258e+01,  4.31042347e+01,
             1.16128480e-04,  1.23185347e+00,  9.10212985e-03,
             0.00000000e+00, -7.38145471e-03,  0.00000000e+00,
             1.00000000e+00, -1.28590163e+00, -3.29769047e-02,
             1.75125265e+02,  1.60426375e+01],
           [            nan,             nan,             nan,
                        nan,             nan,             nan,
                        nan,             nan,             nan,
                        nan,             nan,             nan,
                        nan,             nan],
           [            nan,             nan,             nan,
                        nan,             nan,             nan,
                        nan,             nan,             nan,
                        nan,             nan,             nan,
                        nan,             nan],
           [-5.00057838e+00,  4.12025144e+01,  3.72486536e+01,
             1.23250547e+00,  1.07766403e-04,  1.85889994e-01,
            -1.50959771e-01,  0.00000000e+00,  1.00000000e+00,
             0.00000000e+00, -2.33836243e+01, -5.33149995e-01,
             1.49048715e+01,  4.64404828e+02],
           [-1.88686892e+01,  7.26903047e+01,  8.63348211e+01,
             1.07766403e-04,  1.23027388e+00,  8.95601240e-03,
             0.00000000e+00, -7.27310569e-03,  0.00000000e+00,
             1.00000000e+00, -1.12660195e+00, -2.56866863e-02,
             1.55759080e+02,  1.49048715e+01]])

    Parameters
    ----------
    points3d : np.ndarray
        3D points in format of np.ndarray((n, 3), dtype=float) where n is the
        number of points
    rvec : TYPE
        Rotational vector of the extrinsic parameters
    tvec : TYPE
        Translational vector of the intrinsic parameters
    cmat : TYPE
        3-by-3 camera matrix.
    dvec : TYPE
        distortion coefficients

    Returns
    -------
    imgPoints : TYPE
        the projected points in format of np.ndarray((n, 2), dtype=float)
    jacob : TYPE
        the jacobian matrix in format of np.ndarray((n * 2, 15, dtype=float))

    """
    # create points3d_valid that contains valid points of points3d
    # (assuming some points in points3d are nan points)
    valids, idx_o2n, idx_n2o = validsOfPoints(points3d)
    points3d_valid = points3d[idx_n2o]
    # project only the valid points
    imgPoints_valid, jacob_valid = cv.projectPoints(
        points3d_valid, rvec, tvec, cmat, dvec)
    # create full array of projected points where invalid points remains nan
    nPoints = points3d.shape[0]
    nPoints_valid = points3d_valid.shape[0]
    imgPoints_valid = imgPoints_valid.reshape((-1, 2))
    imgPoints = permuteRows(imgPoints_valid, idx_o2n)
    # jacobian matrix needs to be generated in the same manner, only that 
    # the number of rows of a jacobian is doubled (for example, projecting 
    # 10 points would create a 20-by-m jacobian matrix), so it needs to do 
    # more reshape operations. 
    jacob_valid_tmp = jacob_valid.reshape((nPoints_valid, -1))
    jacob_tmp = permuteRows(jacob_valid_tmp, idx_o2n)
    jacob = jacob_tmp.reshape((2 * nPoints, -1))
    #
    return imgPoints, jacob   
    
    
    
    
    
    
    
    
    
    
    
