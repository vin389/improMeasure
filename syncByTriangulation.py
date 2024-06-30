import time
import datetime
import numpy as np
import scipy
import os
import cv2
from triangulatePoints2 import triangulatePoints2

def syncByTriangulation(xi1, xi2, 
                        syncFrameRange,
                        lagTrials,
                        cmat1, dvec1, rvec1, tvec1, 
                        cmat2, dvec2, rvec2, tvec2,
                        toPlotXi=False,
                        toPlotAllMse=False,
                        toPlotAllPrjErrs=False):
    """
    This function finds the time difference (time lag) between two camera by
    finding which time difference results in the minimum projection errors, 
    given time series of image trajectory of both cameras (xi1 and xi2), the 
    trails of time lags, and camera parameters. 

    Parameters
    ----------
    xi1 : numpy array, (nTimeSteps1, 2), np.float64
        image coordinates of a point in camera 1
    xi2 : numpy array, (nTimeSteps2, 2), np.float64
        image coordinates of a point in camera 2
    syncFrameRange : start and end (end is not included, python style) of 
                     camera 1 frames that user wants to do triangulation. 
    lagTrials : numpy array, (nTrials, ), np.int32
        trials of time lags, e.g., [-10,-8,-6,-4,-2,0,2,4] or
                        np.arange(-10,-5,2)
                        -10 means the camera 2 started 10 frames earlier 
                        than camera 1.
                        4 means the camera 2 started 4 frames later than
                        camera 1. 
    cmat1 : numpy array, (3,3), np.float64
        camera matrix of camera 1
    dvec1 : numpy array, (n,), np.float64
        distortion coefficients of camera 1. The n can be 4, 5, ... 14, 
        depending on how many distortion coefficients are used.
    rvec1 : numpy array, (3,), np.float64
        rotational vector of camera 1
    tvec1 : numpy array, (3,), np.float64
        translational vector of camera 1
    cmat2 : numpy array, (3,3), np.float64
        camera matrix of camera 2
    dvec2 : numpy array, (n,), np.float64
        distortion coefficients of camera 2. The n can be 4, 5, ... 14, 
        depending on how many distortion coefficients are used.
    rvec2 : numpy array, (3,), np.float64
        rotational vector of camera 2
    tvec2 : numpy array, (3,), np.float64
        translational vector of camera 2

    Returns
    -------
    tlagBest : float64 
        the best time lag based on all trials of lagRange.
        tlagBest is optimized by finding the minimum of the parabola around
        the best (minimum) of lagRange trials.
    msePrjErr : numpy array, (lagRange.size, ), np.float64
        the mean-square-error of projection errors of every trials
        For example, if avgPrjErr[0] is 1.234, it means if we set time lag 
          to lagRange[0], and we do triangulation over 
          xi1[t1_start:t1_end] and xi2[t2_start:t2_end]
            where t1_start = max(tLag, 0)
                  t1_end = min(xi1.shape[0], xi2.shape[0] + tLag)
                  t2_start = max(-tLag, 0)
                  t2_end = min(xi2.shape[0], xi1.shape[0] - tLag)
          the average projection error is the average of projection errors
                  of triangulations over t1_start:t1:end (and t2_start:t2_end)
    """
    # reshape and type convertion
    xi1 = xi1.reshape((-1, 2)).astype(np.float64)
    xi2 = xi2.reshape((-1, 2)).astype(np.float64)
    syncFrameRange = np.array(syncFrameRange,dtype=int).flatten()
    lagTrials = np.array(lagTrials,dtype=int).flatten()
    cmat1 = cmat1.reshape(3,3).astype(np.float64)
    dvec1 = dvec1.flatten().astype(np.float64)
    if rvec1.size == 9:
        rvec1 = cv2.Rodrigues(rvec1)[0]
    rvec1 = rvec1.reshape(3,1).astype(np.float64)
    tvec1 = tvec1.reshape(3,1).astype(np.float64)
    cmat2 = cmat2.reshape(3,3).astype(np.float64)
    dvec2 = dvec2.flatten().astype(np.float64)
    if rvec2.size == 9:
        rvec2 = cv2.Rodrigues(rvec2)[0]
    rvec2 = rvec2.reshape(3,1).astype(np.float64)
    tvec2 = tvec2.reshape(3,1).astype(np.float64)
    # plot inputs
    if toPlotXi:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        fig.suptitle("Image coordinates")
        ax.plot(range(xi1.shape[0]), xi1[:,0]-xi1[0,0], label='xi1_ux')
        ax.plot(range(xi1.shape[0]), xi1[:,1]-xi1[0,1], label='xi1_uy')
        ax.plot(range(xi2.shape[0]), xi2[:,0]-xi2[0,0], label='xi2_ux')
        ax.plot(range(xi2.shape[0]), xi2[:,1]-xi2[0,1], label='xi2_uy')
        ax.grid(True); ax.legend();
    # allocate memory
    msePrjErr = np.ones(lagTrials.size, dtype=np.float64) * np.nan
    if toPlotAllPrjErrs:
        import matplotlib.pyplot as plt
        allPrjErrs = np.zeros((4*lagTrials.size, xi1.shape[0]), dtype=np.float32)
    # Check if syncFrameRange is ok. If not okay, adjust syncFrameRange
    if syncFrameRange[0] < 0:
        print("# Warning: syncByTriangulation(): syncFrameRange[0] should be >= 0 but is %d." % syncFrameRange[0])
        syncFrameRange[0] = 0
    if syncFrameRange[1] > xi1.shape[0]:
        print("# Warning: syncByTriangulation(): syncFrameRange[1] should be <= %d but is %d." % (xi1.shape[0], syncFrameRange[1]))
        syncFrameRange[1] = xi1.shape[0]
    if syncFrameRange[0] < max(lagTrials):
        print("# Warning: syncByTriangulation(): syncFrameRange[0] should be >= %d but is %d." % (max(lagTrials), syncFrameRange[0]))
        syncFrameRange[0] = max(lagTrials)
    if syncFrameRange[1] > xi2.shape[0]+min(lagTrials):
        print("# Warning: syncByTriangulation(): syncFrameRange[1] should be <= %d but is %d." % (xi2.shape[0]+min(lagTrials), syncFrameRange[1]))
        syncFrameRange[1] = xi2.shape[0]+min(lagTrials)
    if syncFrameRange[1] <= syncFrameRange[0]:
        print("# Error: syncByTriangulation(): syncFrameRange is too narrow, or lagTrials is too wide.")
        return 

    # run the loop over lagRange
    #   (lagRange could be something like [-120, -118, ..., 118, 120])
    tic_lastPrint = time.time()
    for ilag in range(len(lagTrials)):
        # the time lag, has to be integer. For example, if tLag is 5, 
        # it means the camera 2 started 5 frames later than camera 1. 
        # Triangulation is on xi1[i] and xi2[i - tLag]
        tLag = lagTrials[ilag]
        # the time range for triangulation
        t1_start = syncFrameRange[0]
        t1_end = syncFrameRange[1]
        t2_start = t1_start - tLag
        t2_end = t1_end - tLag
        # do triangulation
        objPoints, objPoints1, objPoints2,\
            prjPoints1, prjPoints2, prjErrors1, prjErrors2 = \
            triangulatePoints2(cmat1, dvec1, rvec1, tvec1, \
                               cmat2, dvec2, rvec2, tvec2, \
                               xi1[t1_start:t1_end, :], \
                               xi2[t2_start:t2_end, :])   
        # calculate the average of projection error      
        errvec = np.concatenate((prjErrors1.flatten(),prjErrors2.flatten())) 
        mse = np.mean(np.square(errvec)) 
        msePrjErr[ilag] = mse
        # store data for plotting all projection error 
        if toPlotAllPrjErrs:
            allPrjErrs[0+ilag*4,t1_start:t1_end] = prjErrors1[:,0]
            allPrjErrs[1+ilag*4,t1_start:t1_end] = prjErrors1[:,1]
            allPrjErrs[2+ilag*4,t1_start:t1_end] = prjErrors2[:,0]
            allPrjErrs[3+ilag*4,t1_start:t1_end] = prjErrors2[:,1]
        # print info every second
        currentTime = time.time()
        if currentTime - tic_lastPrint > 1.0:
            print("# improSync:Finding time lag. Progress:%d/%d. "
                  "Trial lag:%.1f frames. Mean-square-error(MSE)(pixel):%.3f" 
                  % (ilag+1, len(lagTrials), tLag, msePrjErr[ilag]))
            tic_lastPrint = currentTime
    # end of for ilag in lagRange
    # plot all projection errors
    if toPlotAllPrjErrs:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(); 
        fig.suptitle("All Project Errors (unit: pixel)")
        im=ax.imshow(allPrjErrs, cmap='jet'); 
        theColorbar = fig.colorbar(im)
        xlabel = ax.set_xlabel('Frame (of camera 1)')
        ylabel = ax.set_ylabel('Lag trials')
    # plot the synchronization trials
    if toPlotAllMse:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(lagTrials, msePrjErr)
        ax.grid(True)
    # get higher precision of best frame by parabola 
    ilagmin = np.argmin(msePrjErr)
    if ilagmin == 0 or ilagmin == msePrjErr.size-1:
        tlagBest = lagTrials[ilagmin]
    else:
        x0 = lagTrials[ilagmin-1] - lagTrials[ilagmin]
        x1 = 0
        x2 = lagTrials[ilagmin+1] - lagTrials[ilagmin]
        y0 = msePrjErr[ilagmin-1]
        y1 = msePrjErr[ilagmin]
        y2 = msePrjErr[ilagmin+1]
        pmat = np.array([x0**2,x0,1.,x1**2,x1,1.,x2**2,x2,1.]).reshape(3,3)
        pinv = np.linalg.inv(pmat)
        pabc = pinv@(np.array([y0,y1,y2],dtype=np.float64).reshape(3,1))
        pabc = pabc.flatten()
        # d(ax^2+bx+c)/dx=0,2ax+b=0,x=-b/2a
        tlagBest = -.5 * pabc[1] / pabc[0] + lagTrials[ilagmin]
    #
    return tlagBest, msePrjErr
    

if __name__ == '__main__':
    from Camera import Camera
    # synthetic cameras
    #    resolution: 1920 x 1080
    #    fov (x): 120 degrees. (fov y is not set and the aspect ratio is 1)
    #    k1: given. Other coefficients are zeros.
    #    fps: 59.94 fps 
    #    locations: (-5,0,0) and (5,0,0) aiming (0,10,0)
    imgSize = [1920, 1080]
    fovs = 120
    fps = 59.94
    c1 = Camera()
    c1.setCmatByImgsizeFovs(imgSize, fovs);
    c1.dvec[0] = -0.05;
    c1.setRvecTvecByPosAim([-5,0,0], [0,10,0])
    c2 = Camera()
    c2.setCmatByImgsizeFovs(imgSize, fovs);
    c2.dvec[0] = -0.1;
    c2.setRvecTvecByPosAim([5,0,0], [0,10,0])
    
    # synthetic motion
    # motion (coordinates) of target
    nt = 1000
    objPoints = np.zeros((nt, 3), dtype=np.float64)
    radiusX = 3.0
    radiusY = 1.0
    dmp = 0.05
    omega = 2 * np.pi # radian per sec. 
    tt = np.linspace(0, (nt-1) / fps, nt)
    objPoints[:,0] = radiusX * np.cos(omega * tt) * np.exp(-omega*dmp*tt)
    objPoints[:,1] = 10.
    objPoints[:,2] = radiusY * np.sin(omega * tt) * np.exp(-omega*dmp*tt)
    
    # image coordinates of cameras 
    xi1_real, jac = cv2.projectPoints(objPoints, c1.rvec, c1.tvec, c1.cmat, c1.dvec)
    xi1_real = xi1_real.reshape((-1,2))
    xi2_real, jac = cv2.projectPoints(objPoints, c2.rvec, c2.tvec, c2.cmat, c2.dvec)
    xi2_real = xi2_real.reshape((-1,2))
    # time data of camera 1
    tlag1 = 0.5 # lag in unit of second
    nt1 = 800
    t1 = (tt + tlag1)[0:nt1]
    # time data of camera 2
    tlag2 = 0.6 # lag in unit of second
    nt2 = 800
    t2 = (tt + tlag2)[0:nt2]
    # lagged image coordinates in camera 1
    xi1_real_f_x = scipy.interpolate.interp1d(
        tt, xi1_real[:,0], kind='cubic')
    xi1_real_f_y = scipy.interpolate.interp1d(
        tt, xi1_real[:,1], kind='cubic')
    xi1_measure = np.zeros((nt1, 2), dtype=float)
    xi1_measure[:,0] = xi1_real_f_x(t1)    
    xi1_measure[:,1] = xi1_real_f_y(t1)    
    # lagged image coordinates in camera 2
    xi2_real_f_x = scipy.interpolate.interp1d(
        tt, xi2_real[:,0], kind='cubic')
    xi2_real_f_y = scipy.interpolate.interp1d(
        tt, xi2_real[:,1], kind='cubic')
    xi2_measure = np.zeros((nt2, 2), dtype=float)
    xi2_measure[:,0] = xi2_real_f_x(t2)    
    xi2_measure[:,1] = xi2_real_f_y(t2)
    # syncByTriangulation
    syncFrameRange = (100, 700)
    lagTrials = np.linspace(-100., 100, 101)
    tlagsBest, msePrjErr = \
        syncByTriangulation(xi1_measure, xi2_measure,
            syncFrameRange, 
            lagTrials, 
            c1.cmat, c1.dvec, c1.rvec, c1.tvec, 
            c2.cmat, c2.dvec, c2.rvec, c2.tvec,
            toPlotXi=True,
            toPlotAllMse=True,
            toPlotAllPrjErrs=True
            ) 
    #     
    print("Best lag is %f frames, which is %f sec." % (tlagsBest, tlagsBest / fps))
    idealTlag = (tlag2-tlag1) * fps
    print("The ideal answer is %f frames. syncByTriangulation error is %f frames." % (idealTlag, tlagsBest-idealTlag))
    
    
        