import os
import time
import numpy as np
import cv2

from inputs import input2 
from pickTemplates import pickTemplates

def eccTrackVideo(
        videoFilepath=None,
        tmpltFilepath=None,
        tmpltFrameId=0,
        frameRange=None,
        tmpltRange=None,
        mTable=None, # should be (nFrames by 5*nPoints), allocated before calling. 
        saveFilepath=None,
):
    # Create OpenCV VideoCapture object (vid)
    if type(videoFilepath) == str:
        try:
            vid = cv2.VideoCapture(videoFilepath)
        except:
            errMessage = "# Error: eccTrackVideo(): Failed to open video file %s." % videoFilepath
            print(errMessage)
            return (-1, errMessage)
    else:
        errMessage = "# Error: eccTrackVideo(): videoFilepath (str) should be the full path of the video file."
        print(errMessage)
        return (-1, errMessage)
    if vid.isOpened() == False:
        errMessage = "# Error: eccTrackVideo(): Failed to open video file %s." % videoFilepath
        print(errMessage)
        return (-1, errMessage)
    # Print info 
    nFrames = round(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    vWidth = round(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    vHeight = round(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vFps = round(vid.get(cv2.CAP_PROP_FPS))
    print("# eccTrackVideo(): video %s opened. (Height/Width/Frames/Fps)=(%d/%d/%d/%d)." % (videoFilepath, vHeight, vWidth, nFrames, vFps))

    # Get imgInit for template image
    # get initial image for templates (including reading, release, and re-open video)
    for ii in range(tmpltFrameId + 1):
        try:
            ret, imgInit = vid.read()
            imgInit = cv2.cvtColor(imgInit, cv2.COLOR_BGR2GRAY)
        except:
            pass
    vid.release()
    while True:
        vid = cv2.VideoCapture(videoFilepath)
        if vid.isOpened():
            break
        print("# Failed to re-open video %s. Do you want to try again? (Enter n to exit this function)" % videoFilepath)
        if input().strip()[0] == 'n':
            errMessage = "# Error: eccTrackVideo(): videoFilepath (str) cannot be re-opened."
            print(errMessage)
            return (-1, errMessage)
    # 

    # Import templates (from tmpltFilepath) ==> nPoints, tmplts[nPoints, 6]
    try:
        if os.path.exists(tmpltFilepath) == True:
            tmplts = np.loadtxt(tmpltFilepath, delimiter=',')
            nPoints = tmplts.shape[0]
        else:
            # if file does not exist, define by mouse
            print("# Template file %s does not exist. Define it by mouse now." % tmpltFilepath)
            print("  # Enter number of POIs for the video: ")
            nPoints = int(input2())
            results = pickTemplates(
                    img = imgInit, 
                    nPoints=nPoints,
                    savefile=tmpltFilepath, 
                    saveImgfile=os.path.splitext(tmpltFilepath)[0] + '_tmpltPicked' + os.path.splitext(tmpltFilepath)[1])
    except:
        errMessage = "# Error: eccTrackVideo(): Failed to load templates from file %s." % tmpltFilepath
        print(errMessage)
        return (-1, errMessage)
    # print info
    print("# %d templates loaded from %s." % (nPoints, tmpltFilepath))

    # mTable should be pre-allocated. But if it is not, allocate it
    if type(mTable) == type(None):
        mTable = np.ones((nFrames, 5*nPoints), dtype=np.float32) * np.nan

    # Data preparation before tracking
    iFrame = frameRange[0]
    for iPoi in tmpltRange:
        mTable[iFrame, iPoi * 5 + 0] = tmplts[iPoi, 0]
        mTable[iFrame, iPoi * 5 + 1] = tmplts[iPoi, 1]
        mTable[iFrame, iPoi * 5 + 2] = 0.0 # rotation
        mTable[iFrame, iPoi * 5 + 3] = 1.0 # correlation
        mTable[iFrame, iPoi * 5 + 4] = 0.0 # computing time of ECC

    # Start running loop of frames 
    for iFrame in range(nFrames):
        if iFrame >= nFrames:
            break
        # read image
        try:
            ret, frame = vid.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            lastReadFrame = frame.copy()
        except:
            print("# Failed to read video %s frame %d. Skipped." % (videoFilepath, iFrame+1))
            frame = lastReadFrame.copy()
        # skip this frame if it is not within the range
        if iFrame <= frameRange[0]:
            continue
        if iFrame > frameRange[1]:
            break
        # ECC tracking 
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.01)
        for iPoi in tmpltRange:
            if iPoi >= nPoints:
                continue
            # define template
            x0 = round(tmplts[iPoi,2])
            y0 = round(tmplts[iPoi,3])
            x1 = round(x0 + tmplts[iPoi,4])
            y1 = round(y0 + tmplts[iPoi,5])
            dx = tmplts[iPoi, 0] - x0 # the difference between POI and tmplt up-left corner
            dy = tmplts[iPoi, 1] - y0 # the difference between POI and tmplt up-left corner
            tmplt = imgInit[y0:y1, x0:x1].copy()
            guess_x = mTable[iFrame - 1, iPoi*5 + 0] - dx
            guess_y = mTable[iFrame - 1, iPoi*5 + 1] - dy
            guess_r = mTable[iFrame - 1, iPoi*5 + 2] * np.pi / 180.
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
                mTable[iFrame, iPoi*5 + 3] = ret # correlation
            if eccSuccess == False:
                print('# ECC failed at frame %d POI:%d (%s)' % (iFrame+1, iPoi+1, videoFilepath))
                mTable[iFrame, iPoi*5 + 3] = 0.0 # correlation
            toc_ecc1 = time.time()
            mTable[iFrame, iPoi*5  + 4] = toc_ecc1 - tic_ecc1 # ecc time
            # calculates image coordinate and rotation of this POI
            xi0 = np.array([dx, dy, 1.], dtype=np.float32).reshape(3, 1)
            xi1 = warp_matrix @ xi0
            mTable[iFrame, iPoi*5 + 0] = xi1[0,0] # image coord.
            mTable[iFrame, iPoi*5 + 1] = xi1[1,0] # image coord.
            rmat33 = np.eye(3, dtype=warp_matrix.dtype)
            rmat33[0:2,0:2] = warp_matrix[0:2,0:2]
            mTable[iFrame, iPoi*5 + 2] = cv2.Rodrigues(rmat33)[0][2][0] * 180. / np.pi
        # end of for iPoi in tmpltRange
        print('\b'*100, end='')
        print("# Frame %d ECC completed." % (iFrame+1), end='')
    # end of for iFrame in range(frameRange[0], frameRange[1] + 1)
    print('')
    # save mTable 
    if type(saveFilepath) == str:
       np.savetxt(saveFilepath, mTable, delimiter=',')
# end of def eccTrackVideo

if __name__ == '__main__':
    print('# Do you want to run eccTrackVideo demo? (1 to run)')
    toRun = input().strip()
    if toRun == '1':
        videoFilepath = 'D:/ExpDataSamples/20240600-CarletonShakeTableCeilingSystem/preparation_demo10/Cam 2.MP4'
        tmpltFilepath = 'D:/ExpDataSamples/20240600-CarletonShakeTableCeilingSystem/preparation_demo10/templates2.csv'
        tmpltFrameId = 40 # initial frame id
        frameRange = [40, 80]
        tmpltRange = np.arange(0, 20)
        bigTable = np.ones((5000, 600), dtype=np.float32) * np.nan
        mTable = bigTable[:, 1:1+5*20]
        saveFilepath= 'D:/ExpDataSamples/20240600-CarletonShakeTableCeilingSystem/preparation_demo10/test_mTable_v2.csv'
        eccTrackVideo(
            videoFilepath=videoFilepath,
            tmpltFilepath=tmpltFilepath,
            tmpltFrameId=tmpltFrameId,
            frameRange=frameRange,
            tmpltRange=tmpltRange,
            saveFilepath=saveFilepath
        )
    # end of if toRun == '1':


#     global nPoints, prjErr, t_lags
#     # get cameras list (E.g., camsList = [1, 2])
#     camsText = txTrackCamRange.get("0.0", "end").strip()
#     camsList_1base = colonRangeToIntList(camsText)
#     camsList = [num - 1 for num in camsList_1base] # from 1-base to 0-base
#     # get frames list (E.g., framesList = [1, 2, 3, 4, ... 9000])
#     framesText = txTrackFrameRange.get("0.0", "end").strip()
#     framesList_1base = colonRangeToIntList(framesText)
#     framesList = [num - 1 for num in framesList_1base] # from 1-base to 0-base
#     # get points list (E.g., poiList = [1, 2, 3, ... ])
#     poisText = txTrackPointRange.get("0.0", "end").strip()
#     poisList_1base = colonRangeToIntList(poisText)
#     poisList = [num - 1 for num in poisList_1base] # from 1-base to 0-base
#     # get templates (tmplts[icam][iPoi, 0:6] defines the template of point ipoi of camera icam)
#     fTmplts = [None, None]
#     # get file names of videos and templates
#     workDir = txWorkDir.get('0.0', 'end').strip()
#     fVideos[0] = os.path.join(workDir, txVidFile1.get('0.0', 'end').strip())
#     fVideos[1] = os.path.join(workDir, txVidFile2.get('0.0', 'end').strip())
#     fTmplts[0] = os.path.join(workDir, txTmpltFile1.get('0.0', 'end').strip())
#     fTmplts[1] = os.path.join(workDir, txTmpltFile2.get('0.0', 'end').strip())
#     # create video objects
#     for icam in camsList:
#         videos[icam] = cv2.VideoCapture(fVideos[icam])
#         if not videos[icam].isOpened(): 
#             print("# Error opening video file %d" % icam)
#         nFrames[icam] = round(videos[icam].get(cv2.CAP_PROP_FRAME_COUNT))
#     nFrameMax = np.max(nFrames)
#     # Allocate memory for bigTable 
#     bigTable = np.ones((nFrameMax, 600), dtype=np.float32) * np.nan
#     # Initialization before frame-by-frame loop 
#     for icam in camsList:
#         # load initial images
#         tic = time.time()    
#         ret, imgInit = videos[icam].read()
#         imgInit = cv2.cvtColor(imgInit, cv2.COLOR_BGR2GRAY)
#         toc = time.time()
#         imgInits[icam] = imgInit.copy()
#         # image reading time (bigTable columns 106 and 306)
#         bigTable[0, 106 + icam * 200] = toc - tic
#         # load template data
#         if os.path.exists(fTmplts[icam]) == True:
#             tmplts[icam] = np.loadtxt(fTmplts[icam], delimiter=',')
#             nPoints[icam] = tmplts[icam].shape[0]
#         else:
#             # if file does not exist, define by mouse
#             print("# Template file %s does not exist. Define it by mouse now." % fTmplts[icam])
#             print("  # Enter number of POIs for video %d: " % (icam+1))
#             nPoints[icam] = int(input2())
#             results = pickTemplates(
#                     img = imgInits[icam], 
#                     nPoints=nPoints[icam],
#                     savefile=fTmplts[icam], 
#                     saveImgfile=fTmpltImgs[icam])
#     # Data preparation before tracking
#     iFrame = 0
#     bigTable[iFrame, 0] = iFrame + 1 # for user, frame index is 1-based
#     for icam in range(2):
#         for iPoi in poisList:
#             bigTable[iFrame, iPoi * 5 + icam * 200 + 1] = tmplts[icam][iPoi, 0]
#             bigTable[iFrame, iPoi * 5 + icam * 200 + 2] = tmplts[icam][iPoi, 1]
#             bigTable[iFrame, iPoi * 5 + icam * 200 + 3] = 0.0 # rotation
#             bigTable[iFrame, iPoi * 5 + icam * 200 + 4] = 1.0 # correlation
#             bigTable[iFrame, iPoi * 5 + icam * 200 + 5] = 0.0 # computing time of ECC
#     # ECC Tracking
#     #  iFrame is the frame index, starting from 1
#     #  iFrame 1 is the 2nd frame, also the 1st frame to track.
#     #  (iFrame 0 is initial frame, does not need to track)
#     #  Valid frame is from iFrame 1 to iFrame nFrames[icam]
#     #  If videos[icam].read() fails, we use lastReadFrame instead.  
#     #    In demo 10, video 2, the 46th (iFrame=45) and 47th frame (iFrame=46)
#     #    cannot be read, i.e., videos[icam].read() failed and went to 
#     #    exception. The frame image remains the previous one (from a different camera).
#     #    That resulted in the rest of ECC failed to track.
#     lastReadFrame = [imgInits[0].copy(), imgInits[1].copy()]
#     for iFrame in framesList:
#         if iFrame <= 0: # this loop should starts from 1 as frame 0 has been done outside this loop
#             continue
#         if iFrame >= bigTable.shape[0]:
#             break
#         bigTable[iFrame, 0] = iFrame + 1
#         for icam in camsList:
#             # if video has been read over (note: two videos have different lengths, and one of the video is shorter than nFrameMax)
#             if iFrame >= nFrames[icam]:
#                 continue # not "break." If "break" is used, when cam 0 is used up, other cams would be skipped.  
#             # read image
#             try:
#                 tic = time.time()
#                 ret, frame = videos[icam].read()
#                 frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#                 toc = time.time()
#                 bigTable[iFrame, 106 + 200*icam] = toc - tic # t of reading image
#                 lastReadFrame[icam] = frame.copy()
#             except:
#                 print("# Failed to read video %d frame %d. Skipped." % (icam+1, iFrame+1))
#                 frame = lastReadFrame[icam].copy()
#             # ECC tracking 
#             criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.01)
#             for iPoi in poisList:
#                 if iPoi >= nPoints[icam]:
#                     continue
#                 # define template
#                 x0 = round(tmplts[icam][iPoi,2])
#                 y0 = round(tmplts[icam][iPoi,3])
#                 x1 = round(x0 + tmplts[icam][iPoi,4])
#                 y1 = round(y0 + tmplts[icam][iPoi,5])
#                 dx = tmplts[icam][iPoi, 0] - x0 # the difference between POI and tmplt up-left corner
#                 dy = tmplts[icam][iPoi, 1] - y0 # the difference between POI and tmplt up-left corner
#                 tmplt = imgInits[icam][y0:y1, x0:x1].copy()
#                 guess_x = bigTable[iFrame - 1, 1 + iPoi*5 + icam*200 + 0] - dx
#                 guess_y = bigTable[iFrame - 1, 1 + iPoi*5 + icam*200 + 1] - dy
#                 guess_r = bigTable[iFrame - 1, 1 + iPoi*5 + icam*200 + 2] * np.pi / 180.
#                 c = np.cos(guess_r)
#                 s = np.sin(guess_r)
#                 warp_guess = np.array([c, -s, guess_x, s, c, guess_y], dtype=np.float32).reshape(2,3)
#                 # run ECC
#                 tic_ecc1 = time.time()
#                 eccSuccess = False
#                 # Try Eculidean
#                 if eccSuccess == False:
#                     motion_model = cv2.MOTION_EUCLIDEAN
#                     for gaussFiltSize in [0,5,4,6,3,7,8,9]:
#                         try:
#                             if gaussFiltSize <= 0:
#                                 ret, warp_matrix = cv2.findTransformECC(tmplt, frame, warp_guess, motion_model, criteria)
#                             else:
#                                 ret, warp_matrix = cv2.findTransformECC(tmplt, frame, warp_guess, motion_model, criteria, gaussFiltSize=gaussFiltSize)
#                             eccSuccess = True
#                             break
#                         except:
#                             print("# Warning: ECC got exception (Frame %d, Cam %d, Point %d)" % (iFrame+1, icam+1, iPoi+1))
#                             pass
#                 # Try translation (if euclidean failed)
#                 if eccSuccess == False:
#                     motion_model = cv2.MOTION_TRANSLATION
#                     for gaussFiltSize in [0,5,4,6,3,7,8,9]:
#                         try:
#                             if gaussFiltSize <= 0:
#                                 ret, warp_matrix = cv2.findTransformECC(tmplt, frame, warp_guess, motion_model, criteria)
#                             else:
#                                 ret, warp_matrix = cv2.findTransformECC(tmplt, frame, warp_guess, motion_model, criteria, gaussFiltSize=gaussFiltSize)
#                             eccSuccess = True
#                             break
#                         except:
#                             pass
#                 if eccSuccess == True:
#                     bigTable[iFrame, 1 + iPoi*5 + icam*200 + 3] = ret # correlation
#                 if eccSuccess == False:
#                     print('# ECC fails at frame %d cam %d POI:%d' % (iFrame+1, icam+1, iPoi+1))
#                     bigTable[iFrame, 1 + iPoi*5 + icam*200 + 3] = 0.0 # correlation
        
#                 toc_ecc1 = time.time()
#                 bigTable[iFrame, 1 + iPoi*5 + 200*icam + 4] = toc_ecc1 - tic_ecc1 # ecc time
#                 # calculates image coordinate and rotation of this POI
#                 xi0 = np.array([dx, dy, 1.], dtype=np.float32).reshape(3, 1)
#                 xi1 = warp_matrix @ xi0
#                 bigTable[iFrame, 1 + iPoi*5 + 200*icam + 0] = xi1[0,0] # image coord.
#                 bigTable[iFrame, 1 + iPoi*5 + 200*icam + 1] = xi1[1,0] # image coord.
#                 rmat33 = np.eye(3, dtype=warp_matrix.dtype)
#                 rmat33[0:2,0:2] = warp_matrix[0:2,0:2]
#                 bigTable[iFrame, 1 + iPoi*5 + 200*icam + 2] = cv2.Rodrigues(rmat33)[0][2][0] * 180. / np.pi
#             # end of loop of iPoi
#         # end of loop of icam
#         # print info
#         if (iFrame+1) % 10 == 0:
#             print("Frame %d completed ECC." % (iFrame+1) )
#     # end of loop of iFrame
#     # release videos
#     for icam in range(2):
#         if videos[icam].isOpened():
#             videos[icam].release()
#     # 
#     # save bigTable to file
#     event_btSaveData(event)
#     pass
# # end of event_btImageTrack()
