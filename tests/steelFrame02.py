import os, time
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import cv2 as cv

def xcorr(x1, x2):
    """
    Cross-correlation function estimates.
    """
    corr = np.correlate(x1, x2, mode='full')
    lags = np.arange(1 - x1.size, x2.size)
    return corr, lags

def camPos(rvec, tvec):
    """
    This function returns camera position given rvec and tvec
    """
    r44 = np.eye(4, dtype=np.float64)
    if rvec.size == 3:
        r44[0:3,0:3] = cv.Rodrigues(rvec)[0]
    elif rvec.size == 9:
        r44[0:3,0:3] = rvec.reshape((3,3))
    else:
        print("# Warning: camPos: rvec is neither 3x1 or 3x3. Returned zeros.")
        return np.zeros(3, dtype=np.float64)
    r44[0:3,3] = tvec.reshape(3)
    r44inv = np.linalg.inv(r44)
    return r44inv[0:3,3].reshape(3)

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
    objPoints1 = np.zeros((0))
    objPoints2 = np.zeros((0))
    prjPoints1 = np.zeros((0))
    prjPoints2 = np.zeros((0))
    prjErrors1 = np.zeros((0))
    prjErrors2 = np.zeros((0))
    # check 
    nPoints = imgPoints1.shape[0]
    if imgPoints2.shape[0] != nPoints:
        print("# Error: triangulatePoints2(): imgPoints1 and 2 have"
              " different number of points.")
        return objPoints, objPoints1, objPoints2, prjPoints1, prjPoints2, prjErrors1, prjErrors2 
    # memory allocation
    objPoints = np.ones((nPoints, 3), dtype=np.float64) * np.nan
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
    #    
    return objPoints, objPoints1, objPoints2, prjPoints1, prjPoints2, prjErrors1, prjErrors2

def writeDataFrameToExcel(df, xlsFile, sheetName):
    """
    This function writes a pandas DataFrame (df) to a sheet 
    (sheet_name) of an Excel file (xlsFile). If the Excel file
    has been existed, load it befoe writing to keep existing 
    sheets intact. 
    """
    if os.path.exists(xlsFile) == False:
        # if new file
        df.to_excel(xlsFile, sheet_name=sheetName)
    else:
        with pd.ExcelWriter(xlsFile, engine='openpyxl', mode='a',
             if_sheet_exists='overlay') as writer:
            df.to_excel(writer, sheet_name=sheetName)

# Load image data (corrected)
#rootDir = "." # You need to modify the path to yours 
rootDir = r"E:\20221100_SteelFrames\20221129\Analysis_SyncTriang_FujiL_FujiR"
cam1DataFrame = pd.read_excel(
    os.path.join(rootDir, "22PointsMatch.xlsx"),
    sheet_name="Fuji_L_sheet"
)
cam1ImgPoints = np.array(cam1DataFrame.iloc[:,8:44])
#cam1ImgPoints = cam1ImgPoints[:-3,:]  # trim the last 3 rows (because they are nan)

cam2DataFrame = pd.read_excel(
    os.path.join(rootDir, "22PointsMatch.xlsx"),
    sheet_name="Fuji_R_sheet"
)
cam2ImgPoints = np.array(cam2DataFrame.iloc[0:2387,8:44])
#cam2ImgPoints = cam2ImgPoints[:-3,:]  # trim the last 3 rows (because they are nan)

# basic info
nstep1 = cam1ImgPoints.shape[0]
npoint1 = cam1ImgPoints.shape[1] // 2
nstep2 = cam2ImgPoints.shape[0]
npoint2 = cam2ImgPoints.shape[1] // 2
if nstep1 != nstep2 or npoint1 != npoint2:
    print("# Error: The size of two data sheets are not consistent.")
    print("#        (%d, %d) vs. (%d, %d)" % (nstep1, npoint1, nstep2, npoint2))
    # return 
nstep = nstep1
npoint = npoint1

# get major displacement for synchronization 
toPlot = True
point = 1
xy = 1 # xy: x:1 y:2
u1_ori = cam1ImgPoints[:, 2 * (point - 1) + (xy - 1)]
u2_ori = cam2ImgPoints[:, 2 * (point - 1) + (xy - 1)]

# normalization
u1 = u1_ori.copy()
u2 = u2_ori.copy()
t1 = np.linspace(0, u1.size - 1, u1.size)
t2 = np.linspace(0, u1.size - 1, u2.size)
u1 = u1 - np.average(u1); u1 = u1 / np.linalg.norm(u1); 
u2 = u2 - np.average(u2); u2 = u2 / np.linalg.norm(u2); 

if toPlot:
    figU = plt.figure(); 
    figU_ax = figU.add_subplot(111)
    figU_ax.set_title("Normalized Signals")
    figU_ax.grid(True)
    plt.plot(t1, u1, t2, u2); 
    plt.xlabel('Selected Camera Step')
    plt.ylabel('Normalized signal')
    plt.legend(['Signal 1', 'Signal 2'])
    figU.show()

# correlation
corr, lags = xcorr(u1, u2)

# find precise location of max corr   
theArgmax = np.argmax(corr)
if theArgmax - 2 >= 0 and theArgmax + 3 < lags.size:
    xmat = np.zeros((5,3), dtype=float)
    ymat = np.zeros((5,1), dtype=float)
    xmat[:,2] = 1.0
    xmat[:,1] = lags[theArgmax-2:theArgmax+3]
    xmat[:,0] = xmat[:,1] * xmat[:,1]
    ymat[:,0] = corr[theArgmax-2:theArgmax+3]
    xxt = np.matmul(xmat.transpose(), xmat)
    xxtinv = np.linalg.inv(xxt)
    xxtinvxxt = np.matmul(xxtinv, xmat.transpose())
    amat = np.matmul(xxtinvxxt, ymat)
    theLag = -amat[1,0] / (2 * amat[0,0])
    # plot the smoothed correlation curve
    if toPlot:
        lagsm = np.linspace(xmat[0,1], xmat[-1,1], 100)
        corrsm = amat[0] * lagsm * lagsm + amat[1] * lagsm + amat[2]
        figC = plt.figure(); 
        figC_ax = figC.add_subplot(111)
        figC_ax.set_title("Correlation")
        plt.plot(lags, corr, lagsm, corrsm)
        plt.xlabel('Lag step')
        plt.ylabel('Correlation coefficient')
        plt.legend(['Correlation', 'Smoothed'])
        plt.grid(True)
        figC.show()
else:
    theLag = lags[theArgmax]
   

# resampling major displacements by interpolation 
newt2 = t2 + theLag
u2_ori_newf = interp1d(newt2, u2_ori, 
                       kind='linear', 
                       bounds_error=False,
                       fill_value=(u2_ori[0], u2_ori[-1]))
u2_ori_new = u2_ori_newf(t2); 
u2_new = u2_ori_new - np.average(u2_ori_new)
u2_new = u2_new / np.linalg.norm(u2_new)

# plot
if toPlot:
    # plot synchronized original signals
    figU1 = plt.figure(); 
    figU1_ax = figU1.add_subplot(111)
    figU1_ax.plot(t1, u1_ori, t2, u2_ori_new); 
    figU1_ax.set_title("Signals after Sync.")
    figU1_ax.set_xlabel('Selected Camera Step')
    figU1_ax.set_ylabel('Signal')
    figU1_ax.legend(['Signal 1', 'Signal 2 (Sync.)'])
    figU1_ax.grid(True)
    figU1.show()
    # plot synchronized normalized signals
    figU2 = plt.figure(); 
    figU2_ax = figU2.add_subplot(111)
    figU2_ax.plot(t1, u1, t2, u2_new); 
    figU2_ax.set_title("Normalized Signals after Sync.")
    figU2_ax.set_xlabel('Selected Camera Step')
    figU2_ax.set_ylabel('Normalized signal')
    figU2_ax.legend(['Signal 1', 'Signal 2 (Sync.)'])
    figU2_ax.grid(True)
    figU2.show()

# resampling by refinement and interpolation for all points
refinement = 10 # if refinement = 5, time increment is 0.2
nRefStep = 1 + (u1.size -1) * refinement
t_ref = np.linspace(0, u1.size - 1, nRefStep) 
t2_ref = t_ref + theLag
cam1ImgPointsRefSync = np.zeros((t_ref.shape[0], cam1ImgPoints.shape[1]))
cam2ImgPointsRefSync = np.zeros((t_ref.shape[0], cam2ImgPoints.shape[1]))

for i in range(cam1ImgPointsRefSync.shape[1]):
    interp1_newf = interp1d(t1, cam1ImgPoints[:, i], 
                            kind='linear',
                            bounds_error=False,
                            fill_value=(cam1ImgPoints[0,i], 
                                        cam1ImgPoints[-1,i]))
    cam1ImgPointsRefSync[:, i] = interp1_newf(t_ref)
for i in range(cam2ImgPointsRefSync.shape[1]):
    interp1_newf = interp1d(newt2, cam2ImgPoints[:, i], 
                            kind='linear',
                            bounds_error=False,
                            fill_value=(cam2ImgPoints[0,i], 
                                        cam2ImgPoints[-1,i]))
    cam2ImgPointsRefSync[:, i] = interp1_newf(t_ref)

# these steps have big time jumps and will set to nan
nanTimeSteps = [454, 947, 1395, 1567, 2059, 2108]
for i in range(cam2ImgPointsRefSync.shape[0]):
    for j in range(len(nanTimeSteps)):
        if t_ref[i] >= nanTimeSteps[j] - abs(int(theLag + 1)) and \
           t_ref[i] <= nanTimeSteps[j] + abs(int(theLag + 1)):
               cam2ImgPointsRefSync[i,:] = np.nan

toPlotCompareSync = True
if (toPlotCompareSync):
    for i in range(cam2ImgPointsRefSync.shape[1] // 2):
        fig = plt.figure()
        # image coord x of cam 1
        ax = fig.add_subplot(4, 1, 1)
        ax.plot(t1, cam1ImgPoints[:,i * 2], t_ref, 
                cam1ImgPointsRefSync[:, i * 2]); 
        ax.set_title('Cam %d image point %d uix' % (1, i + 1))
        ax.grid(True)
        ax.legend(['Before refinement', 'After refinement'])
        # image coord y of cam 2
        ax = fig.add_subplot(4, 1, 2)
        ax.plot(t1, cam1ImgPoints[:,i * 2 + 1], t_ref, 
                cam1ImgPointsRefSync[:, i * 2 + 1]); 
        ax.set_title('Cam %d image point %d uiy' % (1, i + 1))
        ax.grid(True)
        ax.legend(['Before refinement', 'After refinement'])
        # image coord x of cam 2 
        ax = fig.add_subplot(4, 1, 3)
        ax.plot(t2, cam2ImgPoints[:,i * 2], t_ref, 
                cam2ImgPointsRefSync[:, i * 2]); 
        ax.set_title('Cam %d image point %d uix' % (2, i + 1))
        ax.grid(True)
        ax.legend(['Before sync.', 'After sync'])
        # image coord y of cam 2
        ax = fig.add_subplot(4, 1, 4)
        ax.plot(t2, cam2ImgPoints[:,i * 2 + 1], t_ref, 
                cam2ImgPointsRefSync[:, i * 2 + 1]); 
        ax.set_title('Cam %d image point %d uiy' % (2, i + 1))
        ax.grid(True)
        ax.legend(['Before sync.', 'After sync'])



#
#
cmat1 = np.array([[4.87136363e+03, 0.00000000e+00, 2.97672637e+03],
 [0.00000000e+00, 4.86452155e+03, 2.08836549e+03],
 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]);
dvec1 = np.array([0.00384983, -0.08767134, 0.01839723, -0.00308084, 0.])
rvec1 = np.array([1.28192882, -0.0983159, 0.06644072])
tvec1 = np.array([-4.25902075, 1.99328958, -1.24357172])

cmat2 = np.array([[4.62473057e+03 ,0.00000000e+00 ,3.24657316e+03],
 [0.00000000e+00, 4.71438357e+03, 1.64935209e+03],
 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]); 
dvec2 = np.array([0.11391359, -0.24676436, -0.02839044, 0.01766391, 0.])
rvec2 = np.array([1.10383073, 0.23099647, -0.33094877])
tvec2 = np.array([-8.54156655, 3.2157917,  1.91875225])

# triangulation 
objPointsHist = np.zeros((nRefStep, npoint * 3), dtype=np.float64)
prjPoints1Hist = np.zeros((nRefStep, npoint * 2), dtype=np.float64)
prjErrors1Hist = np.zeros((nRefStep, npoint * 2), dtype=np.float64)
prjPoints2Hist = np.zeros((nRefStep, npoint * 2), dtype=np.float64)
prjErrors2Hist = np.zeros((nRefStep, npoint * 2), dtype=np.float64)

for istep in range(nRefStep):
    imgPoints1 = cam1ImgPointsRefSync[istep,:].reshape(npoint,2)
    imgPoints2 = cam2ImgPointsRefSync[istep,:].reshape(npoint,2)
    objPoints, objPoints1, objPoints2, prjPoints1, prjPoints2, prjErrors1, prjErrors2 =\
        triangulatePoints2(cmat1, dvec1, rvec1, tvec1, \
            cmat2, dvec2, rvec2, tvec2, \
            imgPoints1, imgPoints2
        )
    objPointsHist[istep,:] = objPoints.reshape(1, -1)
    prjPoints1Hist[istep,:] = prjPoints1.reshape(1, -1)
    prjErrors1Hist[istep,:] = prjErrors1.reshape(1, -1)
    prjPoints2Hist[istep,:] = prjPoints2.reshape(1, -1)
    prjErrors2Hist[istep,:] = prjErrors2.reshape(1, -1)

# plot displacements of points 5 to 22 
if toPlot:
    for ip in range(1, 18 + 1):
        figDisp = plt.figure()
        unitFactor = 1000. # m to mm
        # ux
        figDisp_ax = figDisp.add_subplot(311)
        figDisp_ax.set_title('Displacements of P%d' % ip)
        disp = objPointsHist[:, (ip - 1) * 3 + 0].reshape(-1)
        disp = (disp - disp[0]) * unitFactor
        figDisp_ax.plot(t_ref, disp); 
        figDisp_ax.grid(True)
        figDisp_ax.set_ylabel('UX (mm)')
        # uy
        figDisp_ax = figDisp.add_subplot(312)
        disp = objPointsHist[:, (ip - 1) * 3 + 1].reshape(-1)
        disp = (disp - disp[0]) * unitFactor
        figDisp_ax.plot(t_ref, disp); 
        figDisp_ax.grid(True)
        figDisp_ax.set_ylabel('UY (mm)')
        # uz
        figDisp_ax = figDisp.add_subplot(313)
        disp = objPointsHist[:, (ip - 1) * 3 + 2].reshape(-1)
        disp = (disp - disp[0]) * unitFactor
        figDisp_ax.plot(t_ref, disp); 
        figDisp_ax.grid(True)
        figDisp_ax.set_ylabel('UZ (mm)')
        figDisp_ax.set_xlabel('Selected Camera Step')
        #figDisp.show()
        plt.show(block=False)


toPlotPrjErrors = True
if toPlotPrjErrors == True:
    # plot projection error x of points 5 to 22 in camera 1
    for ip in range(1, 18 + 1):
        figPrjErr = plt.figure()
        unitFactor = 1 # pixel 
        # ux
        figPrjErr_ax = figPrjErr.add_subplot(211)
        figPrjErr_ax.set_title('Projection errors of P%d in camera 1' % ip)
        prjErrs = prjErrors1Hist[:, (ip - 1) * 2 + 0].reshape(-1)
        prjErrs = prjErrs * unitFactor
        figPrjErr_ax.plot(t_ref, prjErrs); 
        figPrjErr_ax.grid(True)
        figPrjErr_ax.set_ylabel('Err X (pixel)')
        # uy
        figPrjErr_ax = figPrjErr.add_subplot(212)
        prjErrs = prjErrors1Hist[:, (ip - 1) * 2 + 1].reshape(-1)
        prjErrs = prjErrs * unitFactor
        figPrjErr_ax.plot(t_ref, prjErrs); 
        figPrjErr_ax.grid(True)
        figPrjErr_ax.set_ylabel('Err Y (pixel)')
        figPrjErr_ax.set_xlabel('Selected Camera Step')
        figPrjErr.show()

    # plot projection error x of points 5 to 22 in camera 2
    for ip in range(1, 18 + 1):
        figPrjErr = plt.figure()
        unitFactor = 1 # pixel 
        # ux
        figPrjErr_ax = figPrjErr.add_subplot(211)
        figPrjErr_ax.set_title('Projection errors of P%d in camera 2' % ip)
        prjErrs = prjErrors2Hist[:, (ip - 1) * 2 + 0].reshape(-1)
        prjErrs = prjErrs * unitFactor
        figPrjErr_ax.plot(t_ref, prjErrs); 
        figPrjErr_ax.grid(True)
        figPrjErr_ax.set_ylabel('Err X (pixel)')
        # uy
        figPrjErr_ax = figPrjErr.add_subplot(212)
        prjErrs = prjErrors2Hist[:, (ip - 1) * 2 + 1].reshape(-1)
        prjErrs = prjErrs * unitFactor
        figPrjErr_ax.plot(t_ref, prjErrs); 
        figPrjErr_ax.grid(True)
        figPrjErr_ax.set_ylabel('Err Y (pixel)')
        figPrjErr_ax.set_xlabel('Selected Camera Step')
        figPrjErr.show()
        figPrjErr.show(block=False)

# write to file 
print("# Writing data to Excel file. It takes a few seconds. Please wait.")
ofile = os.path.join(rootDir, 'triangulated.xlsx')
columns3d = []
for i in range(npoint):
    columns3d.append("Xw_%02d " % (i + 1)) 
    columns3d.append("Yw_%02d " % (i + 1)) 
    columns3d.append("Zw_%02d " % (i + 1)) 
columns2d = []
for i in range(npoint):
    columns2d.append("Xw_%02d " % (i + 1)) 
    columns2d.append("Yw_%02d " % (i + 1)) 
#rows = []
#for i in range(nstep):
#    rows.append("%d" % (i + 1))
rows = t_ref.flatten()
# write triangulated points (object points) to file 
df = pd.DataFrame(objPointsHist, columns=columns3d, index=rows)
writeDataFrameToExcel(df, ofile, "triangulated")
# write projected points to file 
df = pd.DataFrame(prjPoints1Hist, columns=columns2d, index=rows)
writeDataFrameToExcel(df, ofile, "Prj.Points.Cam1")
# write projected points to file 
df = pd.DataFrame(prjPoints2Hist, columns=columns2d, index=rows)
writeDataFrameToExcel(df, ofile, "Prj.Points.Cam2")
# write projected errors to file 
df = pd.DataFrame(prjErrors1Hist, columns=columns2d, index=rows)
writeDataFrameToExcel(df, ofile, "Prj.Errors.Cam1")
# write projected errors to file 
df = pd.DataFrame(prjErrors2Hist, columns=columns2d, index=rows)
writeDataFrameToExcel(df, ofile, "Prj.Errors.Cam2")

