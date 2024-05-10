import os
import cv2
import numpy as np
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
            break
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
    
    # Triangulation data preparation
    imgPoints1 = np.zeros((min(nPoints),2), dtype=np.float64)
    imgPoints2 = np.zeros((min(nPoints),2), dtype=np.float64)
    for iPoi in range(min(nPoints)):
        icam = 0
        imgPoints1[iPoi, 0] = bigTable[iFrame, iPoi*5+200*icam+1]
        imgPoints1[iPoi, 1] = bigTable[iFrame, iPoi*5+200*icam+2]
        icam = 1
        imgPoints2[iPoi, 0] = bigTable[iFrame, iPoi*5+200*icam+1]
        imgPoints2[iPoi, 1] = bigTable[iFrame, iPoi*5+200*icam+2]
    cmat1 = cmats[0]; dvec1 = dvecs[0]; rvec1 = rvecs[0]; tvec1 = tvecs[0]
    cmat2 = cmats[1]; dvec2 = dvecs[1]; rvec2 = rvecs[1]; tvec2 = tvecs[1]
    # Triangulation 
    objPoints, objPoints1, objPoints2, \
        prjPoints1, prjPoints2, prjErrors1, prjErrors2 = \
        triangulatePoints2(cmat1, dvec1, rvec1, tvec1, 
                           cmat2, dvec2, rvec2, tvec2, 
                           imgPoints1, imgPoints2)
    # Triangulation post-processing (writing to bigTable)
    for iPoi in range(min(nPoints)):
        bigTable[iFrame, 1 + iPoi*7 + 400] = objPoints[iPoi,0]
        bigTable[iFrame, 1 + iPoi*7 + 401] = objPoints[iPoi,1]
        bigTable[iFrame, 1 + iPoi*7 + 402] = objPoints[iPoi,2]
        bigTable[iFrame, 1 + iPoi*7 + 403] = prjErrors1[iPoi,0]
        bigTable[iFrame, 1 + iPoi*7 + 404] = prjErrors1[iPoi,1]
        bigTable[iFrame, 1 + iPoi*7 + 405] = prjErrors2[iPoi,0]
        bigTable[iFrame, 1 + iPoi*7 + 406] = prjErrors2[iPoi,1]        

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
