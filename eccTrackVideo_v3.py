import os
import time
import numpy as np
import cv2

from inputs import input2 
from pickTemplates import pickTemplates

def eccTrackVideo_v3(
        videoFilepath="",
        tmplts=np.array((0,6),dtype=int),
        tmpltFrameId=0,
        frameRange=np.array([0, 0],dtype=int),
        mTable=np.array([],dtype=np.float64), # should be (nFrames by 6*nPoints), allocated before calling. 
        saveFilepath=None,
):
    # Create OpenCV VideoCapture object (vid)
    try:
        vid = cv2.VideoCapture(videoFilepath)
    except:
        errMessage = "# Error: eccTrackVideo_v3(): Failed to open video file %s." % videoFilepath
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

    # get imgInit, the image that is used to define template images
    for ii in range(tmpltFrameId + 1):
        try:
            ret, imgInit = vid.read()
        except:
            if ii != tmpltFrameId:
                print("# Warning: Failed to read frame %d (1-based index) in video %s. Ignored it." % (ii+1, videoFilepath))
            else:
                errMessage = "# Error: Cannot read the frame for templates (tmpltFrameId) from video."
                return (-1, errMessage)
        # end of try video read
    # end of for ii in range(tmpltFrameid+1):
    imgInit = cv2.cvtColor(imgInit, cv2.COLOR_BGR2GRAY)
    vid.release()
    #
    # Get imgInit for template image
    # get initial image for templates (including reading, release, and re-open video)
    if tmplts.shape[0] <= 0 or tmplts.shape[1] != 6:
        errMessage = "# Warning: eccTrackVideo_v3(): tmplts (templates) should be a N by 6 float array, which N is number of points to track."
        # ask user if they want to define templates by mouse
        print("# Define it by mouse now.")
        print("  # Enter number of POIs for the video: (0 or negative to quit):")
        nPoints = int(input2())
        print("  # Enter full path to save the templates (a %x6 array):" % nPoints)
        tmpltFilepath = input2()
        if nPoints > 0:
            results = pickTemplates(
                    img = imgInit, 
                    nPoints=nPoints,
                    savefile=tmpltFilepath, 
                    saveImgfile=os.path.splitext(tmpltFilepath)[0] + '_tmpltPicked' + os.path.splitext(tmpltFilepath)[1])
        else:
            return (-1, "# User quit")
    # end of if tmplts.shape is invalid
    nPoints = tmplts.shape[0]
    # reopen the video
    while True:
        vid = cv2.VideoCapture(videoFilepath)
        if vid.isOpened():
            break
        print("# Failed to re-open video %s. Do you want to try again? (Enter n to exit this function)" % videoFilepath)
        if input().strip()[0] == 'n':
            errMessage = "# Error: eccTrackVideo(): videoFilepath (str) cannot be re-opened."
            print(errMessage)
            return (-1, errMessage)
    # end of while True that creates VideoCapture

    # mTable should be pre-allocated. But if it is not, allocate it
    if mTable.shape[0] < max(frameRange) or mTable.shape[1] < 6*nPoints:
        errMessage = "# Error: mTable is too small (%s). Should be at least %s." % (str(mTable.shape), str((max(frameRange),6*nPoints)))
        print(errMessage)
        return (-1, errMessage)

    # Data preparation before tracking
    iFrame = frameRange[0]
    for iPoi in range(nPoints):
        mTable[iFrame, iPoi * 6 + 0] = tmplts[iPoi, 0]
        mTable[iFrame, iPoi * 6 + 1] = tmplts[iPoi, 1]
        mTable[iFrame, iPoi * 6 + 2] = 0.0 # rotation
        mTable[iFrame, iPoi * 6 + 3] = 1.0 # correlation
        mTable[iFrame, iPoi * 6 + 4] = 0.0 # computing time of ECC

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
        if iFrame < frameRange[0]:
            continue
        if iFrame >= frameRange[1]:
            break
        # ECC tracking 
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.01)
        for iPoi in range(nPoints):
#            if iPoi >= nPoints:
#                continue
            # define template
            x0 = round(tmplts[iPoi,2])
            y0 = round(tmplts[iPoi,3])
            x1 = round(x0 + tmplts[iPoi,4])
            y1 = round(y0 + tmplts[iPoi,5])
            dx = tmplts[iPoi, 0] - x0 # the difference between POI and tmplt up-left corner
            dy = tmplts[iPoi, 1] - y0 # the difference between POI and tmplt up-left corner
            tmplt = imgInit[y0:y1, x0:x1].copy()
            if iFrame == frameRange[0]:
                guess_x = mTable[iFrame, iPoi*6 + 0] - dx
                guess_y = mTable[iFrame, iPoi*6 + 1] - dy
                guess_r = mTable[iFrame, iPoi*6 + 2] * np.pi / 180.
            else:
                guess_x = mTable[iFrame - 1, iPoi*6 + 0] - dx
                guess_y = mTable[iFrame - 1, iPoi*6 + 1] - dy
                guess_r = mTable[iFrame - 1, iPoi*6 + 2] * np.pi / 180.                
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
                mTable[iFrame, iPoi*6 + 3] = ret # correlation
            if eccSuccess == False:
                print('# ECC failed at frame %d POI:%d (%s)' % (iFrame+1, iPoi+1, videoFilepath))
                mTable[iFrame, iPoi*6 + 3] = 0.0 # correlation
            toc_ecc1 = time.time()
            motion_model
            mTable[iFrame, iPoi*6  + 4] = motion_model
            mTable[iFrame, iPoi*6  + 5] = toc_ecc1 - tic_ecc1 # ecc time
            # calculates image coordinate and rotation of this POI
            xi0 = np.array([dx, dy, 1.], dtype=np.float32).reshape(3, 1)
            xi1 = warp_matrix @ xi0
            mTable[iFrame, iPoi*6 + 0] = xi1[0,0] # image coord.
            mTable[iFrame, iPoi*6 + 1] = xi1[1,0] # image coord.
            rmat33 = np.eye(3, dtype=warp_matrix.dtype)
            rmat33[0:2,0:2] = warp_matrix[0:2,0:2]
            mTable[iFrame, iPoi*6 + 2] = cv2.Rodrigues(rmat33)[0][2][0] * 180. / np.pi
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
    print('# Do you want to run eccTrackVideo demo? (1 to run, other inputs to quit.)')
    toRun = input().strip()
    if toRun == '1':
        videoFilepath = 'D:/ExpDataSamples/20240600-CarletonShakeTableCeilingSystem/preparation_demo10/Cam 2.MP4'
        tmpltFilepath = 'D:/ExpDataSamples/20240600-CarletonShakeTableCeilingSystem/preparation_demo10/templates2.csv'
        tmpltFrameId = 40 # initial frame id
        frameRange = [40, 80]
        tmpltRange = np.arange(0, 20)
        bigTable = np.ones((5000, 600), dtype=np.float32) * np.nan
        mTable = bigTable[:, 1:1+6*20]
        saveFilepath= 'D:/ExpDataSamples/20240600-CarletonShakeTableCeilingSystem/preparation_demo10/test_mTable_v3.csv'
        tmplts = np.loadtxt(tmpltFilepath, delimiter=',')
        eccTrackVideo_v3(
            videoFilepath=videoFilepath,
            tmplts=tmplts,
            tmpltFrameId=tmpltFrameId,
            frameRange=frameRange,
            mTable=bigTable,
            saveFilepath=saveFilepath
        )
    # end of if toRun == 
