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
rootDir = r"D:\ExpDataSamples\20221100_SteelFrames\20221109\Analysis_SyncTriang"
cam1DataFrame = pd.read_excel(
    os.path.join(rootDir, "FujiL_corrected.xlsx"),
    sheet_name="tmatch_picked10Points.mp4_compa"
)
cam1ImgPoints = np.array(cam1DataFrame.iloc[:,26:46])
cam1ImgPoints = cam1ImgPoints[:-3,:]  # trim the last 3 rows (because they are nan)

cam2DataFrame = pd.read_excel(
    os.path.join(rootDir, "SonyR_corrected.xlsx"),
    sheet_name="tmatch_picked10Points.mp4_compa"
)
cam2ImgPoints = np.array(cam2DataFrame.iloc[0:1967,26:46])
cam2ImgPoints = cam2ImgPoints[:-3,:]  # trim the last 3 rows (because they are nan)

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
point = 2
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
    plt.plot(t1, u1, t2, u2); 
    plt.xlabel('Selected Camera Step')
    plt.ylabel('Normalized signal')
    plt.legend(['Signal 1', 'Signal 2'])
    plt.grid(True)
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
   

# resampling by interpolation
u2_ori_newf = interp1d(t2, u2_ori, kind='cubic')
newt2 = t2 - theLag
t2max = np.max(t2)
t2min = np.min(t2)
for i in range(newt2.size): # I think Python has a good way to code this.
    if newt2[i] < t2min:
        newt2[i] = t2min
    if newt2[i] > t2max:
        newt2[i] = t2max
u2_ori_new = u2_ori_newf(newt2); 
u2_new = u2_ori_new - np.average(u2_ori_new)
u2_new = u2_new / np.linalg.norm(u2_new)

# plot
if toPlot:
    # plot synchronized original signals
    figU = plt.figure(); 
    figU_ax = figU.add_subplot(111)
    plt.plot(t1, u1_ori, t2, u2_ori_new); 
    figU_ax.set_title("Signals after Sync.")
    plt.xlabel('Selected Camera Step')
    plt.ylabel('Signal')
    plt.legend(['Signal 1', 'Signal 2 (Sync.)'])
    plt.grid(True)
    figU.show()
    # plot synchronized normalized signals
    figU = plt.figure(); 
    figU_ax = figU.add_subplot(111)
    plt.plot(t1, u1, t2, u2_new); 
    figU_ax.set_title("Normalized Signals after Sync.")
    plt.xlabel('Selected Camera Step')
    plt.ylabel('Normalized signal')
    plt.legend(['Signal 1', 'Signal 2 (Sync.)'])
    plt.grid(True)
    figU.show()

#
#
cmat1 = np.array([[4.96645474e+03, 0.00000000e+00, 3.15001688e+03],
 [0.00000000e+00, 4.94833921e+03, 1.64519766e+03],
 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]);
dvec1 = np.array([ 0.01957185, -0.10239298, -0.02639653, 0.01053248, 0.])
rvec1 = np.array([ 1.19570517, -0.01607302, 0.05885124])
tvec1 = np.array([-5.0925951, 1.42655219, -0.75218652])
cmat2 = np.array([[4.62518280e+03, 0.00000000e+00, 3.58513903e+03],
 [0.00000000e+00, 4.45069734e+03, 1.55297799e+03],
 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]); 
dvec2 = np.array([0.04263902, -0.04625665, -0.02196992,  0.04512054, 0.])
rvec2 = np.array([1.21038212, 0.2140841, -0.36221727])
tvec2 = np.array([-9.60638653, 3.63676106,  2.72955891])

# triangulation 
objPointsHist = np.zeros((nstep, npoint * 3), dtype=np.float64)
prjPoints1Hist = np.zeros((nstep, npoint * 2), dtype=np.float64)
prjErrors1Hist = np.zeros((nstep, npoint * 2), dtype=np.float64)
prjPoints2Hist = np.zeros((nstep, npoint * 2), dtype=np.float64)
prjErrors2Hist = np.zeros((nstep, npoint * 2), dtype=np.float64)

for istep in range(nstep):
    imgPoints1 = cam1ImgPoints[istep,:].reshape(npoint,2)
    imgPoints2 = cam2ImgPoints[istep,:].reshape(npoint,2)
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

# plot displacements of points 2 to 9 
for ip in range(2, 9 + 1):
    figDisp = plt.figure()
    unitFactor = 1000. # m to mm
    # ux
    figDisp_ax = figDisp.add_subplot(311)
    figDisp_ax.set_title('Displacements of P%d' % ip)
    disp = objPointsHist[:, (ip - 1) * 3 + 0].reshape(-1)
    disp = (disp - disp[0]) * unitFactor
    plt.plot(t1, disp); plt.grid(True)
    plt.ylabel('UX (mm)')
    # uy
    figDisp_ax = figDisp.add_subplot(312)
    disp = objPointsHist[:, (ip - 1) * 3 + 1].reshape(-1)
    disp = (disp - disp[0]) * unitFactor
    plt.plot(t1, disp); plt.grid(True)
    plt.ylabel('UY (mm)')
    # uz
    figDisp_ax = figDisp.add_subplot(313)
    disp = objPointsHist[:, (ip - 1) * 3 + 2].reshape(-1)
    disp = (disp - disp[0]) * unitFactor
    plt.plot(t1, disp); plt.grid(True)
    plt.ylabel('UZ (mm)')
    plt.xlabel('Selected Camera Step')
    #figDisp.show()
    plt.show(block=True)


toPlotPrjErrors = False
if toPlotPrjErrors == True:
    # plot projection error x of points 2 to 9 in camera 1
    for ip in range(2, 9 + 1):
        figPrjErr = plt.figure()
        unitFactor = 1 # pixel 
        # ux
        figPrjErr_ax = figPrjErr.add_subplot(211)
        figPrjErr_ax.set_title('Projection errors of P%d in camera 1' % ip)
        prjErrs = prjErrors1Hist[:, (ip - 1) * 2 + 0].reshape(-1)
        prjErrs = prjErrs * unitFactor
        plt.plot(t1, prjErrs); plt.grid(True)
        plt.ylabel('Err X (pixel)')
        # uy
        figPrjErr_ax = figPrjErr.add_subplot(212)
        prjErrs = prjErrors1Hist[:, (ip - 1) * 2 + 1].reshape(-1)
        prjErrs = prjErrs * unitFactor
        plt.plot(t1, prjErrs); plt.grid(True)
        plt.ylabel('Err Y (pixel)')
        plt.xlabel('Selected Camera Step')
        figPrjErr.show()

    # plot projection error x of points 2 to 9 in camera 2
    for ip in range(2, 9 + 1):
        figPrjErr = plt.figure()
        unitFactor = 1 # pixel 
        # ux
        figPrjErr_ax = figPrjErr.add_subplot(211)
        figPrjErr_ax.set_title('Projection errors of P%d in camera 2' % ip)
        prjErrs = prjErrors2Hist[:, (ip - 1) * 2 + 0].reshape(-1)
        prjErrs = prjErrs * unitFactor
        plt.plot(t2, prjErrs); plt.grid(True)
        plt.ylabel('Err X (pixel)')
        # uy
        figPrjErr_ax = figPrjErr.add_subplot(212)
        prjErrs = prjErrors2Hist[:, (ip - 1) * 2 + 1].reshape(-1)
        prjErrs = prjErrs * unitFactor
        plt.plot(t1, prjErrs); plt.grid(True)
        plt.ylabel('Err Y (pixel)')
        plt.xlabel('Selected Camera Step')
        figPrjErr.show()
    #    plt.show(block=True)

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
rows = []
for i in range(nstep):
    rows.append("%d" % (i + 1))
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

