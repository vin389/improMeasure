from math import cos, sin, pi # , sqrt, tan, acos, asin, atan2
import numpy as np
import cv2 as cv
from scipy.optimize import least_squares
import time


def rvecTvecFromR44(r44):
    """
    Returns the rvec and tvec of the camera
    Parameters
    ----------
    r44 : TYPE np.array((4,4),dtype=float)
        The 4-by-4 form of camera extrinsic parameters
    Returns
    -------
    TYPE: tuple ([0]: np.array(3,dtype=float, [1]: np.array(3,dtype=float)))
    Returns the rvec and tvec of the camera 
    """
    rvec, rvecjoc = cv.Rodrigues(r44[0:3,0:3])
    tvec = r44[0:3,3]
    return rvec, tvec

def extrinsicR44ByCamposAndAim(campos, aim):
    """
    Calculates the 4-by-4 matrix form of extrinsic parameters of a camera according to camera position and a point it aims at.
    Considering the world coordinate X-Y-Z where Z is upward, 
    starting from an initial camera orientation (x,y,z) which is (X,-Z,Y), that y is downward (-Z), 
    rotates the camera so that it aims a specified point (aim)
    This function guarantee the camera axis x is always on world plane XY (i.e., x has no Z components)
    Example:
        campos = np.array([ -100, -400, 10],dtype=float)
        aim = np.array([0, -50, 100],dtype=float)
        r44Cam = extrinsicR44ByCamposAndAim(campos,aim)
        # r44Cam would be 
        # np.array([[ 0.961, -0.275,  0.000, -1.374],
        #           [ 0.066,  0.231, -0.971,  108.6],
        #           [ 0.267,  0.933,  0.240,  397.6],
        #           [ 0.000,  0.000,  0.000,  1.000]])
        
    Parameters
    ----------
    campos: TYPE np.array((3,3),dtype=float)
        camera position in the world coordinate 
    aim: TYPE np.array((3,3),dtype=float)
        the aim that the camera is aims at
    Returns
    -------
    TYPE: np.array((4,4),dtype=float)
        the 4-by-4 matrix form of the extrinsic parameters
    """
    # camera vector (extrinsic)
    vz_cam = aim - campos
    vy_cam = np.array([0,0,-1], dtype=np.float64)
    vz_cam = vz_cam / np.linalg.norm(vz_cam)
    vx_cam = np.cross(vy_cam, vz_cam)
    vy_cam = np.cross(vz_cam, vx_cam)
    vx_cam = vx_cam / np.linalg.norm(vx_cam)
    vy_cam = vy_cam / np.linalg.norm(vy_cam)
    vz_cam = vz_cam / np.linalg.norm(vz_cam)
    r44inv = np.eye(4, dtype=np.float64)
    r44inv[0:3, 0] = vx_cam[0:3]
    r44inv[0:3, 1] = vy_cam[0:3]
    r44inv[0:3, 2] = vz_cam[0:3]
    r44inv[0:3, 3] = campos[0:3]
    r44 = np.linalg.inv(r44inv)
    return r44


def bladePointByThetaAndDeflection(r_blade, r44_blade, theta, deflection): 
    """
    calculates the 3D coordinates of blade point 
    Example: 
        # the radius of the point on the blade is 55 meters
        r_blade = 55.0; 
        # blade faces towards -Y of global coord. (i.e., blade z is 0,-1,0)
        r44_blade = np.array([[-1,0,0,0],[0,0,-1,0],[0,-1,0,0],[0,0,0,1]],dtype=float)
        # blade rotates to the highest point
        theta = -90; 
        # deflection is 0.3 m along blade z axis
        deflection = 0.3
        # get the coordinate of peak (peakPoint)
        peakPoint = bladePeakByThetaAndDeflection(r_blade, r44_blade, theta, deflection)
    
    Parameters
    ----------
    r_blade : TYPE float
        radius of the blade point
    r44_blade : TYPE numpy.ndarray ((4,4), dtype=np.float)
        the extrinsic parameters of the blade in 4-by-4 matrix form
    theta : TYPE float 
        the angle (in degree) of the blade on the blade axes x and y (note: the blade y axis could be commonly downward. Think carefully about the blade axes according to the r44_blade)
    deflection : TYPE float
        the deflection of the blade along blade axis z 
    
    Returns
    -------
    TYPE (4, dtype=np.float) 
    Returns the 3D coordinate of the point (homogeneous coordinate).
    """
    bladePointLocal = np.array([r_blade * cos(theta / (180./pi)), \
                      r_blade * sin(theta / (180./pi)), deflection, 1], \
                      dtype=np.float64)
    r44inv_blade = np.linalg.inv(r44_blade)  
    bladePointGlobal = np.matmul(r44inv_blade, bladePointLocal.transpose())
    bladePointGlobal /= bladePointGlobal[3]
    return bladePointGlobal


def bladeImgPointByThetaAndDeflection(theta, deflection, r_blade, r44_blade, cmat, dvec, r44_cam):
    """
    calculates the image coordinates of blade point 
    Parameters
    ----------
    theta : TYPE float 
        the angle (in degree) of the blade on the blade axes x and y (note: the blade y axis could be commonly downward. Think carefully about the blade axes according to the r44_blade)
    deflection : TYPE float
        the deflection of the blade along blade axis z 
    r_blade : TYPE float 
        radius of the blade point
    r44_blade : TYPE numpy.ndarray ((4,4), dtype=np.float)
        the extrinsic parameters of the blade in 4-by-4 matrix form
    cmat : TYPE numpy.ndarray((3,3), dtype=np.float)
        camera matrix 
    dvec : TYPE numpy.ndarray(n, dtype=np.float)
        distortion vector
    r44_cam : TYPE numpy.ndarray((4,4),dtype=np.float)
        extrinsic parameters
    Returns
    -------
    TYPE (2, dtype=np.float) 
    Returns the image coordinate of the point .
    """
    bladeWorldPoint = bladePointByThetaAndDeflection(r_blade, r44_blade, theta, deflection)
    rvec, tvec = rvecTvecFromR44(r44_cam)
    bladeImagePoint, jacob = cv.projectPoints(bladeWorldPoint[0:3],                                               
                                              rvec, tvec, cmat, dvec)    
    bladeImagePoint = bladeImagePoint.reshape(2)
    return bladeImagePoint

def funBladeImgPointByThetaAndDeflection(x, r_blade, r44_blade, cmat, dvec, r44_cam, imgPoint):
    theta = x[0]
    deflection = x[1]   
    bladeImagePoint = bladeImgPointByThetaAndDeflection(theta, deflection, r_blade, r44_blade, cmat, dvec, r44_cam)   
    bladeImagePoint -= imgPoint 
    return bladeImagePoint

def bladeThetaAndDeflectionByImgPoint(bladeImagePoint, r_blade, r44_blade, cmat, dvec, r44_cam):
    minCost = 1e30
    bestTheta = -1;
    bestDeflection = -1;
    bestRes =[];
    for theta_i in range(10):
        initTheta = theta_i * 36.0 
        x0 = np.array((initTheta, 0),dtype=float)
        lbound = np.array([  0.0, -r_blade * 0.2])
        ubound = np.array([360.0, +r_blade * 0.2])
        #lbound = np.array([  0.0, -r_blade * 0.1])
        #ubound = np.array([360.0, +r_blade * 0.1])
        bounds = (lbound, ubound)
        res_lsq = least_squares(funBladeImgPointByThetaAndDeflection, x0, \
            bounds= bounds, 
            args=(r_blade, r44_blade, cmat, dvec, r44_cam, bladeImagePoint))
        if (res_lsq.cost < minCost):
            minCost = res_lsq.cost
            bestTheta = res_lsq.x[0]
            bestDeflection = res_lsq.x[1]
            bestRes = res_lsq
#        print(res_lsq)
#        print('-----------------------')
    eigs = np.linalg.eig(bestRes.jac)
    condition = max(abs(eigs[0])) / min(abs(eigs[0]))
    if condition > 10:
        print(bestTheta, bestDeflection, condition) 
    return bestTheta, bestDeflection, condition


def bladeThetaAndDeflectionByImgPoint2(bladeImagePoint, r_blade, r44_blade, cmat, dvec, r44_cam):
    minCost = 1e30
    bestTheta = -1;
    bestDeflection = -1;
    bestRes =[];
    rvec = cv.Rodrigues(r44_cam[0:3,0:3])[0]
    tvec = r44_cam[0:3,3].reshape(3,1)
    # find the angle (bestTheta) which project point is close to 
    # the given image point, assuming the preliminary bestDeflection 
    # is zero
    nTrialTheta = 60
    for theta_i in range(nTrialTheta):
        initTheta = theta_i * 360. / nTrialTheta
        x_tip_turbCoord = np.array([
            r_blade * cos(initTheta * np.pi / 180.),
            r_blade * sin(initTheta * np.pi / 180.),
            0.0, 1.], dtype=float).reshape((4,1))
        x_tip_world = np.linalg.inv(r44_blade) @ x_tip_turbCoord
        x_tip_img = cv.projectPoints(x_tip_world[0:3], 
                                     rvec, tvec, cmat, dvec)[0].reshape(-1)
        cost = np.linalg.norm(bladeImagePoint.flatten() - x_tip_img.flatten())
        if cost < minCost:
            minCost = cost
            bestTheta = initTheta
            bestDeflection = 0.0
    # run least square method to find the best theta and deflection, 
    # given the initial guess (bestTheta, 0.0)
    # and the upper/lower bounds:
    #    bestTheta +/- 30 degrees
    #    bestDeflection +/- r_blade * 0.2
    x0 = np.array((bestTheta, bestDeflection),dtype=float)
    lbound = np.array([bestTheta - 20., -r_blade * 0.2])
    ubound = np.array([bestTheta + 20., +r_blade * 0.2])
    bounds = (lbound, ubound)
    res_lsq = least_squares(funBladeImgPointByThetaAndDeflection, x0,
        bounds= bounds, 
        args=(r_blade, r44_blade, cmat, dvec, r44_cam, bladeImagePoint))
    minCost = res_lsq.cost
    bestTheta = res_lsq.x[0]
    bestDeflection = res_lsq.x[1]
    bestRes = res_lsq
    # find the condition number of the system (jacobian matrix)
    eigs = np.linalg.eig(bestRes.jac)
    condition = max(abs(eigs[0])) / min(abs(eigs[0]))
    # return best theta and deflection, and the system condition number 
    return bestTheta, bestDeflection, condition


if __name__ == '__main__':
    #    import matplotlib.pyplot as plt
    from draw3dMesh import draw3dMesh
    from drawPoints import drawPoints
    # Turbine
    rBlade = 55.; # meter
    turbineCenter = np.array([100, 150, 120], dtype=float)
    turbineAim = turbineCenter + np.array([0, -10, 0])
#    turbineAim = np.array([0,-300,10])
    r44Turbine = extrinsicR44ByCamposAndAim(turbineCenter, turbineAim)
    rpm = -10.0 # 10.0 round per minute
    vibFreq = 2.7 # vibration frequency of blade
    ampDeflection = 5.0
    print ('# The turbine center is at ', turbineCenter)
    print ('# The turbine aims towards ', turbineAim)
    print("# The R44 matrix of turbine is:\n", r44Turbine)
    print("# The rotating speed is %f RPM." % rpm)
    print("# The deflection amplification is %f m." % ampDeflection)
    # camera 
    cameraCenter = np.array([0, -300, 10], dtype=float)
    cameraAim = turbineCenter
    r44Camera = extrinsicR44ByCamposAndAim(cameraCenter, cameraAim)
    rvecCamera = cv.Rodrigues(r44Camera[0:3,0:3])[0]
    tvecCamera = r44Camera[0:3,3].reshape(3,1)
    imgSize = np.array([600,800], dtype=int)
#    cmat = np.array([1111.1111, 0, 400, 0, 1111.1111, 300, 0,0,1]).reshape(3,3)
    cmat = np.array([2222.2222, 0, 400, 0, 2222.2222, 300, 0,0,1]).reshape(3,3)
    dvec = np.array([0,0,0,0,0.]).reshape(-1,1)
    fps = 60.0 # 60 frames per second
    print ('# The camera center is at ', cameraCenter)
    print ('# The camera aims towards ', cameraAim)
    print("# The R44 matrix of camera is:\n", r44Camera)
    print("# Camera parameters:")
    print("#  Camera matrix:\n", cmat)
    print("#  Distortion coefficients: ", dvec.flatten())
    print("# Recording speed: %f FPS (or Hz)." % fps)

    # trajectory
    nRound = 0.99
    nPoint = int(nRound / abs(rpm) * 60. * fps + 0.5) 
    xtip_imgPoints = np.zeros((nPoint, 2), dtype=float)
    noise = 1.0 # noise of trajectory image point in unit of pixels
    print("# Noise of image point: %.2f pixels" % noise)
    # initial image to draw the trajectory
    imgH, imgW = imgSize[0], imgSize[1]
    img = np.zeros((imgH, imgW, 3), dtype=np.uint8)
    img = draw3dMesh(img, cmat, dvec, rvecCamera, tvecCamera, 
               meshx=np.linspace(50,150,11), 
               meshy=np.array([150.]), 
               meshz=np.linspace(70,170,11),
               thickness=1, color=(0,128,0), savefile='.')
    computingTime = 0.0
    countComputingTime = 0
    # analyzed data table 
    #   [:,0]: time
    #   [:,1]: ground truth of theta
    #   [:,2]: ground truth of deflection
    #   [:,3]: analyzed theta
    #   [:,4]: analyzed deflection
    #   [:,5]: analyzed system condition number
    #   [:,6]: error of theta (analyzed - ground truth)
    #   [:,7]: error of deflection (analyzed - ground truth)
    analyzedData = np.ones((nPoint, 8), dtype=float) * np.nan
    # loop
    for i in range(nPoint):
        t = i * 1. / fps
        theta_deg = -90 + t * rpm / 60. * 360.
        theta_deg = theta_deg % 360.0
        #deflection = 0.0
        deflection = ampDeflection * sin(2. * np.pi * vibFreq * t)
        # calculate image points (forward)
        xtip_imgPoints[i,:] = bladeImgPointByThetaAndDeflection(
            theta_deg, deflection, rBlade, r44Turbine, cmat, dvec, r44Camera)        
        # add noise
        xtip_imgPoints[i,:] += np.random.rand(2) * noise
        # calculate deflection (backward)
        tic = time.time()
        bestTheta, bestDeflection, condition =\
            bladeThetaAndDeflectionByImgPoint2(
                xtip_imgPoints[i,:], rBlade, r44Turbine, cmat, dvec, r44Camera)
        toc = time.time()
        computingTime += toc - tic
        countComputingTime += 1
        # analyzed data table
        analyzedData[i, 0] = t
        analyzedData[i, 1] = theta_deg
        analyzedData[i, 2] = deflection
        analyzedData[i, 3] = bestTheta
        analyzedData[i, 4] = bestDeflection
        analyzedData[i, 5] = condition
        analyzedData[i, 6] = bestTheta - theta_deg
        if analyzedData[i, 6] > 180:
            analyzedData[i, 6] -= 360.
        if analyzedData[i, 6] < -180:
            analyzedData[i, 6] += 360.
        analyzedData[i, 7] = bestDeflection - deflection

        #   draw
        drawPoints(img, xtip_imgPoints[i,:], color=(255,255,255), markerSize=3, savefile='.')
        cv.imshow('Blade Trajectory', img)
        ikey = cv.waitKey(1); 
        if ikey > 0:
            break
    print("Computing time %.6f sec. per point" 
          % (computingTime / countComputingTime))

    cv.waitKey(0)
    try:
        cv.destroyWindow('Blade Trajectory')
    except:
        pass
    
    # statistics
    errTheta = analyzedData[:, 6].copy()
    errDefle = analyzedData[:, 7].copy()
    conditionNumThreshold = [2., 4., 8., 16., 32., 64., 128., 256.]
    for icond in range(len(conditionNumThreshold)):
        count = 0
        sumErrTheta = 0.
        sumErrDefle = 0.
        ssqErrTheta = 0.
        ssqErrDefle = 0.
        for i in range(nPoint):
            if analyzedData[i, 5] <= conditionNumThreshold[icond]:
                count += 1
                sumErrTheta += errTheta[i]
                ssqErrTheta += errTheta[i] ** 2
                sumErrDefle += errDefle[i]
                ssqErrDefle += errDefle[i] ** 2
        stdErrTheta = ((ssqErrTheta / count) - (sumErrTheta / count) ** 2) ** .5
        stdErrDefle = ((ssqErrDefle / count) - (sumErrDefle / count) ** 2) ** .5
        print("\n\n# Considering noise of %.2f pixels:" % noise)
        print("# For condition number < %.1f:" % conditionNumThreshold[icond])
        print("#   Count/Total: %d/%d" % (count, nPoint))
        print("#   Std of theta      error: %.2f" % stdErrTheta)
        print("#   Std of deflection error: %.2f" % stdErrDefle)
