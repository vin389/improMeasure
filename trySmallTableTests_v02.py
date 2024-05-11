import os
import cv2
import numpy as np
import scipy
import time 
#from imshow2 import imshow2
from pickTemplates import pickTemplates
from inputs import input2
from triangulatePoints2 import triangulatePoints2

t_veryBegin = time.time()

# video files
fVideos = [None, None] 
fVideos[0] = r'D:\ExpDataSamples\20240600-CarletonShakeTableCeilingSystem\preparation_demo10\Cam 1.mp4'
fVideos[1] = r'D:\ExpDataSamples\20240600-CarletonShakeTableCeilingSystem\preparation_demo10\Cam 2.mp4'
# template data files
fTmplts = [None, None]
fTmplts[0] = 'D:\\ExpDataSamples\\20240600-CarletonShakeTableCeilingSystem\\preparation_demo10\\templates1.csv'
fTmplts[1] = 'D:\\ExpDataSamples\\20240600-CarletonShakeTableCeilingSystem\\preparation_demo10\\templates2.csv'
# template image files
fTmpltImgs = [None, None]
fTmpltImgs[0] = 'D:\\ExpDataSamples\\20240600-CarletonShakeTableCeilingSystem\\preparation_demo10\\templates1.jpg'
fTmpltImgs[1] = 'D:\\ExpDataSamples\\20240600-CarletonShakeTableCeilingSystem\\preparation_demo10\\templates2.jpg'
# calibration parameters files
fCalibs = [None, None]
fCalibs[0] = r'D:\ExpDataSamples\20240600-CarletonShakeTableCeilingSystem\preparation_demo10\cam1_parameters.csv'
fCalibs[1] = r'D:\ExpDataSamples\20240600-CarletonShakeTableCeilingSystem\preparation_demo10\cam2_parameters.csv'
# big table files
fBigTable = r'D:\ExpDataSamples\20240600-CarletonShakeTableCeilingSystem\preparation_demo10\bigTables.csv'

# Initialization before frame-by-frame loop 
# Including: 
#   Load videos
#     --> videos[]
#   
videos = [None, None]
nFrames = [0, 0]
nPoints = [0, 0]
imgInits = [None, None]
tmplts = [None, None]
cmats = [None, None]
dvecs = [None, None]
rvecs = [None, None]
tvecs = [None, None]

for icam in range(2):
    # load videos
    videos[icam] = cv2.VideoCapture(fVideos[icam])
    if not videos[icam].isOpened(): 
        print("# Error opening video file %d" % icam)
    nFrames[icam] = round(videos[icam].get(cv2.CAP_PROP_FRAME_COUNT))
nFrameMax = np.max(nFrames)

# Allocate memory for bigTable 
bigTable = np.ones((nFrameMax, 600), dtype=np.float32) * np.nan

# Initialization before frame-by-frame loop 
# Including: 
#   Load initial images (i.e., iFrame is 0) for template definition
#     --> imgInits[]
#   Load template coordinates
#     --> tmplts[]
#   Load camera parameters
#     --> cmats[], dvecs[], rvecs[], tvecs[]
#   
for icam in range(2):
    # Load initial images
    tic = time.time()    
    ret, imgInit = videos[icam].read()
    imgInit = cv2.cvtColor(imgInit, cv2.COLOR_BGR2GRAY)
    toc = time.time()
    imgInits[icam] = imgInit.copy()

    # image reading time (bigTable columns 106 and 306)
    bigTable[0, 106 + icam * 200] = toc - tic

    # load template data
    if os.path.exists(fTmplts[icam]) == True:
        tmplts[icam] = np.loadtxt(fTmplts[icam], delimiter=',')
        nPoints[icam] = tmplts[icam].shape[0]
    else:
        # if file does not exist, define by mouse
        print("# Template file %s does not exist. Define it by mouse now." % fTmplts[icam])
        print("  # Enter number of POIs for video %d: " % (icam+1))
        nPoints[icam] = int(input2())
        results = pickTemplates(
                img = imgInits[icam], 
                nPoints=nPoints[icam],
                savefile=fTmplts[icam], 
                saveImgfile=fTmpltImgs[icam])
           
    # load calibration parameters
    try:
        camParam = np.loadtxt(fCalibs[icam], delimiter=',')
        rvecs[icam] = camParam[2:5]
        tvecs[icam] = camParam[5:8]
        cmats[icam] = camParam[8:17].reshape(3,3)
        dvecs[icam] = camParam[17:]
    except:
        print("# Failed to load camera calibration file %s" % fCalibs[icam])

# Data preparation before tracking
iFrame = 0
bigTable[iFrame, 0] = iFrame + 1 # for user, frame index is 1-based
for icam in range(2):
    for iPoi in range(nPoints[icam]):
        bigTable[iFrame, iPoi * 5 + icam * 200 + 1] = tmplts[icam][iPoi, 0]
        bigTable[iFrame, iPoi * 5 + icam * 200 + 2] = tmplts[icam][iPoi, 1]
        bigTable[iFrame, iPoi * 5 + icam * 200 + 3] = 0.0 # rotation
        bigTable[iFrame, iPoi * 5 + icam * 200 + 4] = 1.0 # correlation
        bigTable[iFrame, iPoi * 5 + icam * 200 + 5] = 0.0 # computing time of ECC

# ECC Tracking
#  iFrame is the frame index, starting from 1
#  iFrame 1 is the 2nd frame, also the 1st frame to track.
#  (iFrame 0 is initial frame, does not need to track)
#  Valid frame is from iFrame 1 to iFrame nFrames[icam]
#  If videos[icam].read() fails, we use lastReadFrame instead.  
#    In demo 10, video 2, the 47th (iFrame=46) and 48th frame (iFrame=47)
#    cannot be read, i.e., videos[icam].read() failed and went to 
#    exception. The frame image remains the previous one (that is, the 47th of
#    video 1). That resulted in the rest of ECC failed to track.
lastReadFrame = [imgInits[0].copy(), imgInits[1].copy()]
for iFrame in range(1, nFrameMax):
    bigTable[iFrame, 0] = iFrame + 1
    for icam in range(2):
        # if video has been read over (note: two videos have different lengths, and one of the video is shorter than nFrameMax)
        if iFrame >= nFrames[icam]:
            continue # not "break." If "break" is used, when cam 0 is used up, other cams would be skipped.  
        # read image
        try:
            tic = time.time()
            ret, frame = videos[icam].read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            toc = time.time()
            bigTable[iFrame, 106 + 200*icam] = toc - tic # t of reading image
            lastReadFrame[icam] = frame.copy()
        except:
            print("# Failed to read video %d frame %d. Skipped." % (icam+1, iFrame+1))
            frame = lastReadFrame[icam].copy()
        # ECC tracking 
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.01)
        for iPoi in range(max(nPoints)):
            if iPoi >= nPoints[icam]:
                break
            # define template
            x0 = round(tmplts[icam][iPoi,2])
            y0 = round(tmplts[icam][iPoi,3])
            x1 = round(x0 + tmplts[icam][iPoi,4])
            y1 = round(y0 + tmplts[icam][iPoi,5])
            dx = tmplts[icam][iPoi, 0] - x0 # the difference between POI and tmplt up-left corner
            dy = tmplts[icam][iPoi, 1] - y0 # the difference between POI and tmplt up-left corner
            tmplt = imgInits[icam][y0:y1, x0:x1].copy()
#            guess_x = bigTable1[iFrame - 1, i * 2 + 1] # it is incorrect
#            guess_y = bigTable1[iFrame - 1, i * 2 + 2] # it is incorrect
            guess_x = bigTable[iFrame - 1, 1 + iPoi*5 + icam*200 + 0] - dx
            guess_y = bigTable[iFrame - 1, 1 + iPoi*5 + icam*200 + 1] - dy
            guess_r = bigTable[iFrame - 1, 1 + iPoi*5 + icam*200 + 2] * np.pi / 180.
            c = np.cos(guess_r)
            s = np.sin(guess_r)
            warp_guess = np.array([c, -s, guess_x, s, c, guess_y], dtype=np.float32).reshape(2,3)
            # run ECC
            tic_ecc1 = time.time()
            eccSuccess = False
            # Try Eculidean
            if eccSuccess == False:
                motion_model = cv2.MOTION_EUCLIDEAN
                for gaussFiltSize in [0,5,4,6,3,7,8,9]:
                    try:
                        if gaussFiltSize <= 0:
                            ret, warp_matrix = cv2.findTransformECC(tmplt, frame, warp_guess, motion_model, criteria)
                        else:
                            ret, warp_matrix = cv2.findTransformECC(tmplt, frame, warp_guess, motion_model, criteria, gaussFiltSize=gaussFiltSize)
                        eccSuccess = True
                        break
                    except:
                        pass
            # Try translation (if euclidean failed)
            if eccSuccess == False:
                motion_model = cv2.MOTION_TRANSLATION
                for gaussFiltSize in [0,5,4,6,3,7,8,9]:
                    try:
                        if gaussFiltSize <= 0:
                            ret, warp_matrix = cv2.findTransformECC(tmplt, frame, warp_guess, motion_model, criteria)
                        else:
                            ret, warp_matrix = cv2.findTransformECC(tmplt, frame, warp_guess, motion_model, criteria, gaussFiltSize=gaussFiltSize)
                        eccSuccess = True
                        break
                    except:
                        pass
            if eccSuccess == True:
                bigTable[iFrame, 1 + iPoi*5 + icam*200 + 3] = ret # correlation
            if eccSuccess == False:
                print('# ECC fails at frame %d cam %d POI:%d' % (iFrame+1, icam+1, iPoi+1))
                bigTable[iFrame, 1 + iPoi*5 + icam*200 + 3] = 0.0 # correlation
     
            toc_ecc1 = time.time()
            bigTable[iFrame, 1 + iPoi*5 + 200*icam + 4] = toc_ecc1 - tic_ecc1 # ecc time

            # calculates image coordinate and rotation of this POI
            xi0 = np.array([dx, dy, 1.], dtype=np.float32).reshape(3, 1)
            xi1 = warp_matrix @ xi0
            bigTable[iFrame, 1 + iPoi*5 + 200*icam + 0] = xi1[0,0] # image coord.
            bigTable[iFrame, 1 + iPoi*5 + 200*icam + 1] = xi1[1,0] # image coord.
            rmat33 = np.eye(3, dtype=warp_matrix.dtype)
            rmat33[0:2,0:2] = warp_matrix[0:2,0:2]
            bigTable[iFrame, 1 + iPoi*5 + 200*icam + 2] = cv2.Rodrigues(rmat33)[0][2][0] * 180. / np.pi

            # print disp
#            print('Frame:%4d, POI:%2d, OK:%1d, Gs:%1d Xi(%6.1f,%6.1f)' % (iFrame, iPoi, eccSuccess, gaussFiltSize, xi1[0,0], xi1[1,0]))
        # end of loop iPoi()        
    # end of loop of icam    
    
    # plot / visualization (use matplotlib canvas renderer)
    tic_plot = time.time()
    toPlot = True
    # only do plotting for iFrame 1, the very last iFrame, 
    # and every 20 frames
    if toPlot == True and (iFrame == 1 or iFrame == nFrameMax - 1 or iFrame % 20 == 1):
        try:
            # for first frame, do initial setting
            if iFrame == 1:
                import matplotlib.pyplot as plt
                import matplotlib.style as mplstyle
                mplstyle.use('fast')
                try:
                    # turn off interactive mode so that it does not 
                    # popup a window, as we want to plot on canvas (in memory)
                    plt.ioff()
                except:
                    pass
                plt.rcParams['figure.dpi'] = 100 # 100 DPI
            # create 20 subplots (4 by 5). entire figure size is 16-inch by 20-inch 
            fig, axes = plt.subplots(4, 5, figsize=(16, 8))  # figsize in inch
            fig.suptitle("Xi movement (Progress: %d/%d)" % (iFrame + 1, nFrameMax))
            for iPoi in range(20):  
                nFrame1tmp = nFrameMax
                xlim = (0, nFrame1tmp)
                ylim = (-20, 20)
                xdata = np.linspace(1, nFrame1tmp, nFrame1tmp)
                ydata = bigTable[0:nFrame1tmp, 1 + 5 * iPoi] - bigTable[0, 1 + 5 * iPoi]
                ax = axes.flatten()[iPoi]
                ax.plot(xdata, ydata, '.', markersize=1)
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
            # Render figure into an image
            fig.canvas.draw()
            img_plot = np.array(fig.canvas.renderer.buffer_rgba())
            img_plot = cv2.cvtColor(img_plot, cv2.COLOR_RGBA2BGR)
            cv2.imshow("Plot", img_plot)
            cv2.waitKey(1)
            # for iFrame < (nFrame1 - 1) clear fig memory            
            # keep the very last plot so that we can save
            # it or other operation
            if iFrame < (nFrameMax - 1):
                plt.close(fig)
        except:
            print("# Cannot plot")
            pass
    # fill bigTable data    
    toc_plot = time.time()
    bigTable[iFrame, 107] = toc_plot - tic_plot # t of reading image
    bigTable[iFrame, 108] = time.time() - t_veryBegin

#    if iFrame>nFrame1tmp:
#        break

# save bigTable to file
np.savetxt(fBigTable, bigTable, delimiter=",")

# save fig
#try:
#    import pickle
#    pickle.dump(fig, open(figFile1, 'wb'))  
#    # Use fig = pickle.load(open('myfigfile.p', 'rb'))
#except:
#    print("# Failed to save figure to a file.")

# close window and release v1     
try:
    cv2.waitKey(0)
    cv2.destroyWindow("Plot")
except:
    pass

for icam in range(2):
    if videos[icam].isOpened():
        videos[icam].release()



# determine the trial t_lag 
t_interest_range = [6*60, 66*60]  # triangulation time range in video 1 (in unit of frame, have to be integers)
t_lag_bounds = [-8*60, -5*60]  # possible time lag of video 2
#t_interest_range = [0*60, 66*60]  # triangulation time range in video 1 (in unit of frame, have to be integers)
#t_lag_bounds = [0,0]  # possible time lag of video 2
t_lags = np.linspace(min(t_lag_bounds), max(t_lag_bounds), 
                     round(abs(t_lag_bounds[1]-t_lag_bounds[0])+1))

# check 
# t_interest[0] - t_lag_bounds[1] should be >= 0.0 
# That is, right cam may have a time lag up to t_lag_bounds[1], 
# and if t_interest[0] is too small, right cam would miss 
# the beginning part of the video time of interests.
t_interest_range_forcedToChange = False
t_interest_range_original = t_interest_range.copy()
if t_interest_range[0] - t_lag_bounds[1] < 0.0:
    t_interest_range[0] = int(t_lag_bounds[1] + 1) 
    t_interest_range_forcedToChange = True
# t_v1_valid[1] - t_lag_bounds[0] should be < (nFrames[1]-1)
# That is, right cam may have a minimum t_lag_bounds[0],
# and if nFrames[1]-1 is not long enough, right cam would miss 
# the ending part of the video time of interests.
if t_interest_range[1] - t_lag_bounds[0] >= (nFrames[1]-1):
    t_interest_range[1] = int(nFrames[1]-1 + t_lag_bounds[0]) 
    t_interest_range_forcedToChange = True
if t_interest_range_forcedToChange:
    print("# Warning: t_lag_bounds is changed from (%d,%d) to (%d,%d)" %
          (t_interest_range_original[0], t_interest_range_original[1], 
           t_interest_range[0],          t_interest_range[1]))
# t_interests[icam]: time tags of frames of interests 
#   For example, if you are only interested in the triangulation of 
#   frames 300 to 3900 in cam 0, t_interest = [300, 301, ..., 3900]
#   and t_v2_interest = [330, 331, 332., ...] (if t_lags is 30)
nFrameInterest = round(t_interest_range[1] - 
                       t_interest_range[0])
t_interest = np.linspace(t_interest_range[0], 
                         t_interest_range[1],
                         nFrameInterest)

imgPoints1_xyi_thisPoi_interest = np.zeros((t_interest.size, 2), dtype=np.float64)
imgPoints2_xyi_thisPoi_interest = np.zeros((t_interest.size, 2), dtype=np.float64)

prjErr = np.zeros((min(nPoints), t_lags.size), dtype=float)

ticMeasures = np.zeros(100)
for ilag in range(t_lags.size):
    tic = time.time()
    t_lag = t_lags[ilag]
    # t_v1 and t_v2: based on the t_tag, set time tags of every frame   
    # t_v1[] would be [0., 1., 2., ...]
    # t_v2[] would be [30., 31., 32., ...] if video 2 lags 30 seconds (turned on 30s later)    
    t_v1 = np.linspace(0, nFrames[0] - 1, round(nFrames[0]))
    t_v2 = np.linspace(0, nFrames[1] - 1, round(nFrames[1])) + t_lag
    ticMeasures[0] += time.time() - tic    
    
#    for iPoi in range(1):
    for iPoi in range(min(nPoints)):
        # print info
        # image points of this point in cam 0
        tic = time.time()
        icam = 0
        xi = bigTable[0:t_v1.size, iPoi*5+200*icam+1]
        yi = bigTable[0:t_v1.size, iPoi*5+200*icam+2]
        imgPoints1_xyi_thisPoi_interest[:,0] = xi[
            round(t_interest[0]):round(t_interest[-1])].copy()
        imgPoints1_xyi_thisPoi_interest[:,1] = yi[
            round(t_interest[0]):round(t_interest[-1])].copy()
        ticMeasures[1] += time.time() - tic    


        # image points of this point in cam 1
        # create interpolation objects for cam 2 (v2)
        tic = time.time()
        
        icam = 1
        xi = bigTable[0:t_v2.size, iPoi*5+200*icam+1]
        yi = bigTable[0:t_v2.size, iPoi*5+200*icam+2]
        # in case xi has nan, fill all nan in xi with its previous value
        if np.isnan(xi[0]):
            xi[0] = 0.0
            print("# Warning: Cam 2 Point %d starts from nan. " % (iPoi))
        for iiFrame in range(xi.size):
            if np.isnan(xi[iiFrame]):
                xi[iiFrame] = xi[iiFrame - 1]
        if np.isnan(yi[0]):
            yi[0] = 0.0
            print("# Warning: Cam 2 Point %d starts from nan. " % (iPoi))
        for iiFrame in range(yi.size):
            if np.isnan(yi[iiFrame]):
                yi[iiFrame] = yi[iiFrame - 1]
        ticMeasures[2] += time.time() - tic    

        tic = time.time()
        intpv2xi = scipy.interpolate.interp1d(t_v2, xi, kind='cubic')
        intpv2yi = scipy.interpolate.interp1d(t_v2, yi, kind='cubic')
        imgPoints2_xyi_thisPoi_interest[:,0] = intpv2xi(t_interest)
        imgPoints2_xyi_thisPoi_interest[:,1] = intpv2yi(t_interest)
        ticMeasures[3] += time.time() - tic    

        # plot for debug
        debugPlotSync = False
        if debugPlotSync and ilag == 0 and iPoi == 0:
            plt.plot(t_interest, imgPoints1_xyi_thisPoi_interest[:,0] - imgPoints1_xyi_thisPoi_interest[:,0][0], '.-', markersize=3, label='Cam 1')
            plt.plot(t_interest, imgPoints2_xyi_thisPoi_interest[:,0] - imgPoints2_xyi_thisPoi_interest[:,0][0], '.-', markersize=3, label='Cam 2')
            plt.legend()
            plt.show()
        # Triangulation
        tic = time.time()
        objPoints, objPoints1, objPoints2, \
            prjPoints1, prjPoints2, prjErrors1, prjErrors2 = \
            triangulatePoints2(cmats[0], dvecs[0], rvecs[0], tvecs[0], 
                               cmats[1], dvecs[1], rvecs[1], tvecs[1], 
                               imgPoints1_xyi_thisPoi_interest, 
                               imgPoints2_xyi_thisPoi_interest)
        ticMeasures[4] += time.time() - tic    

        # debug
        debugPlotPrjErrors = False
        if debugPlotPrjErrors and ilag == 0 and iPoi == 0:
            fig, axes = plt.subplots(nrows=4, ncols=1, sharex=True)
            fig.suptitle('Projection errors (in pixel)')
            axes[0].plot(t_interest, prjErrors1[:,0], '.-', markersize=3, label='c1x'); axes[0].grid('True')
            axes[1].plot(t_interest, prjErrors1[:,1], '.-', markersize=3, label='c1y'); axes[1].grid('True')
            axes[2].plot(t_interest, prjErrors2[:,0], '.-', markersize=3, label='c2x'); axes[2].grid('True')
            axes[3].plot(t_interest, prjErrors2[:,1], '.-', markersize=3, label='c2y'); axes[3].grid('True')
        # norm of error
        tic = time.time()
        err = (np.linalg.norm(prjErrors1.flatten()) + np.linalg.norm(prjErrors2.flatten())) / np.sqrt(2*prjErrors1.shape[0])
        prjErr[iPoi, ilag] = err
        ticMeasures[5] += time.time() - tic    
        # 
        # print info
        print("# If time lag is %.3fs, average projection error of POI %d is %.3f" % (t_lag, iPoi+1, err) )

t_lag_p = np.zeros(min(nPoints), dtype=float)
for iPoi in range(min(nPoints)):
    # find the t_lag which results in minimum prjErr[iPoi, :]
    ilagmin = np.argmin(prjErr[iPoi,:])
    t2 = t_lags[ilagmin]
    t1 = t2 - 1
    t3 = t2 + 1
    e1 = prjErr[iPoi, ilagmin - 1]
    e2 = prjErr[iPoi, ilagmin]
    e3 = prjErr[iPoi, ilagmin + 1]
    # shift time axis so that t2_ = 0, t2_ = t2 - t2
    tmat = np.array([1, -1, 1, 0, 0, 1, 1, 1, 1], dtype=float).reshape(3,3)
    evec = np.array([e1, e2, e3], dtype=float).reshape(3, 1)
    coef = np.linalg.inv(tmat) @ evec
    t_lag_p[iPoi] = -.5 * coef[1] / coef[0] + t2

plt.plot(range(1,21), t_lag_p, '.-')
plt.title('Best time lags (frame) ')
plt.xlabel('Point')
plt.ylabel('Best time lag (frame)')
plt.grid(True)









