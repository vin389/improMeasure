import os, sys
import time, datetime
import tkinter as tk
import tkinter.filedialog
import tkinter.messagebox
import numpy as np
import scipy
import cv2


from improCalib import newArrayByMapping, countCalibPoints
from improMisc import uigetfile, uiputfile, uigetfiles_tupleFullpath
from improStrings import npFromString, stringFromNp, npFromTupleNp
from drawPoints import drawPoints
from imshow2 import imshow2
from writeCamera import writeCamera
from draw3dMesh import draw3dMesh
from inputs import input2
from pickTemplates import pickTemplates
from triangulatePoints2 import triangulatePoints2

global win, bigTable, fVideos, fTmplts, fTmpltImgs, fCalibs, videos, nFrames, nPoints, imgInits, tmplts, cmats, dvecs, rvecs, tvecs
global prjErr
fVideos = [None, None] 
fTmplts = [None, None]
fTmpltImgs = [None, None]
fCalibs = [None, None]
videos = [None, None]
nFrames = [0, 0]
nPoints = [0, 0]
imgInits = [None, None]
tmplts = [None, None]
cmats = [None, None]
dvecs = [None, None]
rvecs = [None, None]
tvecs = [None, None]


def tkStereosync():
    global win
    win = tk.Tk()
    win.title("Stereo synchronization (v.0.20240524)")
    win.geometry("1400x700")
    
    # Frame 1 about working directory
    frame1 = tk.Frame(win, highlightbackground="gray", highlightthickness=1)
    frame1.pack(fill=tk.X)
    # Button btWorkDir
    btWorkDir = tk.Button(frame1, width=40, text="Working directory")
    btWorkDir.pack(side=tk.LEFT, padx=5, pady=2)
    # Text txWorkDir
    txWorkDir = tk.Text(frame1, width=100, height=1, undo=True, 
                        autoseparators=True, maxundo=-1)
    txWorkDir.pack(side=tk.LEFT, padx=5, pady=2)

    # Frame 2 about videos
    frame2 = tk.Frame(win, highlightbackground="gray", highlightthickness=1)
    frame2.pack(fill=tk.X)
    lbAboutVid = tk.Label(frame2, text="About videos", fg='gray')
    lbAboutVid.pack(side=tk.TOP, anchor=tk.W)
    # Button btVidFile1
    btVidFile1 = tk.Button(frame2, width=30, text="Video file 1")
    btVidFile1.pack(side=tk.LEFT, padx=5, pady=2)
    # Text txVidFile1
    txVidFile1 = tk.Text(frame2, width=50, height=1, undo=True, 
                        autoseparators=True, maxundo=-1)
    txVidFile1.pack(side=tk.LEFT, padx=5, pady=2)
    # Button btVidFile2
    btVidFile2 = tk.Button(frame2, width=30, text="Video file 2")
    btVidFile2.pack(side=tk.LEFT, padx=5, pady=2)
    # Text txVidFile2
    txVidFile2 = tk.Text(frame2, width=50, height=1, undo=True, 
                        autoseparators=True, maxundo=-1)
    txVidFile2.pack(side=tk.LEFT, padx=5, pady=2)

    # Frame 3 about calibration
    frame3 = tk.Frame(win, highlightbackground="gray", highlightthickness=1)
    frame3.pack(fill=tk.X)
    lbAboutCalibration = tk.Label(frame3, text="About calibration", fg='gray')
    lbAboutCalibration.pack(side=tk.TOP, anchor=tk.W)
    # Button btCalibFile1
    btCalibFile1 = tk.Button(frame3, width=30, text="Calibration file 1")
    btCalibFile1.pack(side=tk.LEFT, padx=5, pady=2)
    # Text txCalibFile1
    txCalibFile1 = tk.Text(frame3, width=50, height=1, undo=True, 
                        autoseparators=True, maxundo=-1)
    txCalibFile1.pack(side=tk.LEFT, padx=5, pady=2)
    # Button btCalibFile2
    btCalibFile2 = tk.Button(frame3, width=30, text="Calibration file 2")
    btCalibFile2.pack(side=tk.LEFT, padx=5, pady=2)
    # Text txCalibFile2
    txCalibFile2 = tk.Text(frame3, width=50, height=1, undo=True, 
                        autoseparators=True, maxundo=-1)
    txCalibFile2.pack(side=tk.LEFT, padx=5, pady=2)

    # Frame 4 about templates
    frame4 = tk.Frame(win, highlightbackground="gray", highlightthickness=1)
    frame4.pack(fill=tk.X)
    lbAboutTmplts = tk.Label(frame4, text="About tracking points (templates)", fg='gray')
    lbAboutTmplts.pack(side=tk.TOP, anchor=tk.W)
    # Button btTmpltFile1
    btTmpltFile1 = tk.Button(frame4, width=30, text="Templates file 1")
    btTmpltFile1.pack(side=tk.LEFT, padx=5, pady=2)
    # Text txCalibText1
    txTmpltFile1 = tk.Text(frame4, width=50, height=1, undo=True, 
                        autoseparators=True, maxundo=-1)
    txTmpltFile1.pack(side=tk.LEFT, padx=5, pady=2)
    # Button btCalibFile1
    btTmpltFile2 = tk.Button(frame4, width=30, text="Templates file 2")
    btTmpltFile2.pack(side=tk.LEFT, padx=5, pady=2)
    # Text txCalibFile2
    txTmpltFile2 = tk.Text(frame4, width=50, height=1, undo=True, 
                        autoseparators=True, maxundo=-1)
    txTmpltFile2.pack(side=tk.LEFT, padx=5, pady=2)

    # Frame 5 about tracking
    frame5 = tk.Frame(win, highlightbackground="gray", highlightthickness=1)
    frame5.pack(fill=tk.X, pady=2)
    lbAboutTracking = tk.Label(frame5, text="About tracking", fg='gray')
    lbAboutTracking.grid(row=0, column=0, sticky="W")
    # Label lbTrackCamRange
    lbTrackCamRange = tk.Label(frame5, width=30, text="Range of tracking cameras")
    lbTrackCamRange.grid(row=1, column=0, padx=5, pady=2)
    # Text txTrackCamRange
    txTrackCamRange = tk.Text(frame5, width=50, height=1, undo=True, 
                        autoseparators=True, maxundo=-1)
    txTrackCamRange.grid(row=1, column=1, padx=5, pady=2)
    # Label lbTrackPointRange
    lbTrackPointRange = tk.Label(frame5, width=30, text="Range of tracking points")
    lbTrackPointRange.grid(row=2, column=0, padx=5, pady=2)
    # Text txTrackPointRange
    txTrackPointRange = tk.Text(frame5, width=50, height=1, undo=True, 
                        autoseparators=True, maxundo=-1)
    txTrackPointRange.grid(row=2, column=1, padx=5, pady=2)
    # Label lbTrackFrameRange
    lbTrackFrameRange = tk.Label(frame5, width=30, text="Range of tracking frames")
    lbTrackFrameRange.grid(row=3, column=0, padx=5, pady=2)
    # Text txTrackFrameRange
    txTrackFrameRange = tk.Text(frame5, width=50, height=1, undo=True, 
                        autoseparators=True, maxundo=-1)
    txTrackFrameRange.grid(row=3, column=1, padx=5, pady=2)
    # Button btImageTrack
    btImageTrack = tk.Button(frame5, width=30, text="Image Track")
    btImageTrack.grid(row=3, column=2, padx=5, pady=2)

    # Frame 6 about synchronization analysis
    frame6 = tk.Frame(win, highlightbackground="gray", highlightthickness=1)
    frame6.pack(fill=tk.X, pady=2)
    lbAboutSync = tk.Label(frame6, text="About synchronization analysis", fg='gray')
    lbAboutSync.grid(row=0, column=0, sticky="W")
    # Label lbPossibleLag
    lbPossibleLag = tk.Label(frame6, width=30, text="Range of possible lag")
    lbPossibleLag.grid(row=1, column=0, padx=5, pady=2)
    # Text txPossibleLag
    txPossibleLag = tk.Text(frame6, width=50, height=1, undo=True, 
                        autoseparators=True, maxundo=-1)
    txPossibleLag.grid(row=1, column=1, padx=5, pady=2)
    # Label lbSyncPoiRange
    lbSyncPoiRange = tk.Label(frame6, width=30, text="Range of points for sync analysis")
    lbSyncPoiRange.grid(row=2, column=0, padx=5, pady=2)
    # Text txSyncPoiRange
    txSyncPoiRange = tk.Text(frame6, width=50, height=1, undo=True, 
                        autoseparators=True, maxundo=-1)
    txSyncPoiRange.grid(row=2, column=1, padx=5, pady=2)
    # Label lbSyncFrameRange
    lbSyncFrameRange = tk.Label(frame6, width=30, text="Range of frames for sync analysis")
    lbSyncFrameRange.grid(row=3, column=0, padx=5, pady=2)
    # Text txSyncFrameRange
    txSyncFrameRange = tk.Text(frame6, width=50, height=1, undo=True, 
                        autoseparators=True, maxundo=-1)
    txSyncFrameRange.grid(row=3, column=1, padx=5, pady=2)
    # Button btFindTimeLags
    btFindTimeLags = tk.Button(frame6, width=30, text="Find best time lags (each point)")
    btFindTimeLags.grid(row=1, column=2, padx=5, pady=2)
    # Button btLags
    lbLags = tk.Label(frame6, width=30, text="Time lags (each point)")
    lbLags.grid(row=2, column=2, padx=5, pady=2)
    # Text txLags
    txLags = tk.Text(frame6, width=50, height=1, undo=True, 
                        autoseparators=True, maxundo=-1)
    txLags.grid(row=2, column=3, padx=5, pady=2)

    # Frame 7 about triangulation
    frame7 = tk.Frame(win, highlightbackground="gray", highlightthickness=1)
    frame7.pack(fill=tk.X, pady=2)
    lbAboutTriang = tk.Label(frame7, text="About triangulation", fg='gray')
    lbAboutTriang.grid(row=0, column=0, sticky="W")
    # Label lbTriangPoiRange
    lbTriangPoiRange = tk.Label(frame7, width=30, text="Range of triangulation points")
    lbTriangPoiRange.grid(row=1, column=0, padx=5, pady=2)
    # Text txTriangPoiRange
    txTriangPoiRange = tk.Text(frame7, width=50, height=1, undo=True, 
                        autoseparators=True, maxundo=-1)
    txTriangPoiRange.grid(row=1, column=1, padx=5, pady=2)
    # Label lbTriangFrameRange
    lbTriangFrameRange = tk.Label(frame7, width=30, text="Range of triangulation frames")
    lbTriangFrameRange.grid(row=2, column=0, padx=5, pady=2)
    # Text txTriangFrameRange
    txTriangFrameRange = tk.Text(frame7, width=50, height=1, undo=True, 
                        autoseparators=True, maxundo=-1)
    txTriangFrameRange.grid(row=2, column=1, padx=5, pady=2)
    # Button Triangulation
    btTriang = tk.Button(frame7, width=30, text="Triangulation")
    btTriang.grid(row=1, column=2, padx=5, pady=2)
    
    # Frame 8 about load and save 
    frame8 = tk.Frame(win, highlightbackground="gray", highlightthickness=1)
    frame8.pack(fill=tk.X, pady=2)
    lbAboutSave = tk.Label(frame8, text="About loading/saving files", fg='gray')
    lbAboutSave.grid(row=0, column=0, sticky="W")
    # Button btLoadConfig
    btLoadConfig = tk.Button(frame8, width=30, text="Load config")
    btLoadConfig.grid(row=1, column=0, padx=5, pady=2)
    # Text txLoadConfig
    txLoadConfig = tk.Text(frame8, width=50, height=1, undo=True, 
                        autoseparators=True, maxundo=-1)
    txLoadConfig.grid(row=1, column=1, padx=5, pady=2)
    txLoadConfig.delete('0.0', 'end')
    txLoadConfig.insert('0.0', 'Stereosync_config.xml')
    # Button btLoadData
    btLoadData = tk.Button(frame8, width=30, text="Load data")
    btLoadData.grid(row=2, column=0, padx=5, pady=2)
    # Text txLoadData
    txLoadData = tk.Text(frame8, width=50, height=1, undo=True, 
                        autoseparators=True, maxundo=-1)
    txLoadData.grid(row=2, column=1, padx=5, pady=2)
    txLoadData.delete('0.0', 'end')
    txLoadData.insert('0.0', 'Stereosync_data.csv')
    # Button btSaveConfig
    btSaveConfig = tk.Button(frame8, width=30, text="Save config")
    btSaveConfig.grid(row=1, column=2, padx=5, pady=2)
    # Text txSaveConfig
    txSaveConfig = tk.Text(frame8, width=50, height=1, undo=True, 
                        autoseparators=True, maxundo=-1)
    txSaveConfig.grid(row=1, column=3, padx=5, pady=2)
    txSaveConfig.delete('0.0', 'end')
    txSaveConfig.insert('0.0', 'Stereosync_config.xml')
    # Button btSaveData
    btSaveData = tk.Button(frame8, width=30, text="Save data")
    btSaveData.grid(row=2, column=2, padx=5, pady=2)
    # Text txSaveConfig
    txSaveData = tk.Text(frame8, width=50, height=1, undo=True, 
                        autoseparators=True, maxundo=-1)
    txSaveData.grid(row=2, column=3, padx=5, pady=2)
    txSaveData.delete('0.0', 'end')
    txSaveData.insert('0.0', 'Stereosync_data.csv')

    # bind event of btWorkDir
    def event_btWorkDir(event):
        tmpwin = tk.Tk()
        tmpwin.lift()    
        workDir = tk.filedialog.askdirectory(title="Select the working directory")
        tmpwin.destroy()
        # quit if user cancels the file dialog
        if type(workDir) == type('') and len(workDir) == 0:
            print("You cancelled the directory dialog.")
            return
        # display workDir to txWorkDir 
        txWorkDir.delete("1.0", "end")
        txWorkDir.insert("1.0", workDir)
        # load config
        loadConfig()
    btWorkDir.bind("<ButtonRelease-1>", event_btWorkDir)

    ####################################
    # bind event of btVidFile1, btVidFile2, btCalibFile1, btCalibFile2, btTmpltFile1, btTmpltFile2
    # as they have similar behaviors, we use a general function event_btFileDialogAndShow()
    ####################################
    def event_btFileDialogAndShow(event, title, display):
        tmpwin = tk.Tk()
        tmpwin.lift()
        initialdir = txWorkDir.get("1.0", "end").strip()
        if os.path.isdir(initialdir) == False:
            initialdir = "c:/"
        vfile = tk.filedialog.askopenfilename(title=title,
                                              initialdir=initialdir)
        tmpwin.destroy()
        # quit if user cancels the file dialog
        if type(vfile) == type('') and len(vfile) == 0:
            print("You cancelled the file dialog.")
            return
        # display file to display widget 
        if os.path.dirname(vfile) == txWorkDir.get("1.0", "end").strip():
            display.delete("1.0", "end")
            display.insert("1.0", os.path.basename(vfile))
        else:
            display.delete("1.0", "end")
            display.insert("1.0", vfile)
            
    # When user clicks "Video file 1" or "Video file 2"
    btVidFile1.bind("<ButtonRelease-1>", lambda event:
                    event_btFileDialogAndShow(event,
                      title="Select the video file of video 1",
                      display=txVidFile1))
    btVidFile2.bind("<ButtonRelease-1>", lambda event:
                    event_btFileDialogAndShow(event,
                      title="Select the video file of video 2",
                      display=txVidFile2))
    # When user clicks "Calibration file 1" or "Calibration file 2"
    btCalibFile1.bind("<ButtonRelease-1>", lambda event:
                    event_btFileDialogAndShow(event,
                      title="Select the calibration file of video 1",
                      display=txCalibFile1))
    btCalibFile2.bind("<ButtonRelease-1>", lambda event:
                    event_btFileDialogAndShow(event,
                      title="Select the calibration file of video 2",
                      display=txCalibFile2))
    # When user clicks "Templates file 1" or "Templates file 2"
    btTmpltFile1.bind("<ButtonRelease-1>", lambda event:
                    event_btFileDialogAndShow(event,
                      title="Select the templates file of video 1",
                      display=txTmpltFile1))
    btTmpltFile2.bind("<ButtonRelease-1>", lambda event:
                    event_btFileDialogAndShow(event,
                      title="Select the templates file of video 2",
                      display=txTmpltFile2))
    
    # From colon-separate integer range to list of integer 
    # Range is in Matlab style, not Python style.
    # That is, "2:5" mean 2 3 4 5,　not 2 3 4. 
    # For example, colonRangeToIntList("2:5  10  15  17:20") would be [2, 3, 4, 5, 10, 15, 17, 18, 19, 20]
    def colonRangeToIntList(theStr):
        sequences = []
        for part in theStr.split():
            try:
                # Try converting the part to an integer (single number)
                num = int(part)
                sequences.append(num)
            except ValueError:
                # If conversion fails, assume it's a colon-separated range
                start, end = map(int, part.split(":"))
                sequences.extend(range(start, end+1))
        return sequences

    # bind event of btImageTrack ("Image Track")
    def event_btImageTrack(event):
        global nPoints, prjErr, t_lags
        # get cameras list (E.g., camsList = [1, 2])
        camsText = txTrackCamRange.get("0.0", "end").strip()
        camsList_1base = colonRangeToIntList(camsText)
        camsList = [num - 1 for num in camsList_1base] # from 1-base to 0-base
        # get frames list (E.g., framesList = [1, 2, 3, 4, ... 9000])
        framesText = txTrackFrameRange.get("0.0", "end").strip()
        framesList_1base = colonRangeToIntList(framesText)
        framesList = [num - 1 for num in framesList_1base] # from 1-base to 0-base
        # get points list (E.g., poiList = [1, 2, 3, ... ])
        poisText = txTrackPointRange.get("0.0", "end").strip()
        poisList_1base = colonRangeToIntList(poisText)
        poisList = [num - 1 for num in poisList_1base] # from 1-base to 0-base
        # get templates (tmplts[icam][iPoi, 0:6] defines the template of point ipoi of camera icam)
        fTmplts = [None, None]
        # get file names of videos and templates
        workDir = txWorkDir.get('0.0', 'end').strip()
        fVideos[0] = os.path.join(workDir, txVidFile1.get('0.0', 'end').strip())
        fVideos[1] = os.path.join(workDir, txVidFile2.get('0.0', 'end').strip())
        fTmplts[0] = os.path.join(workDir, txTmpltFile1.get('0.0', 'end').strip())
        fTmplts[1] = os.path.join(workDir, txTmpltFile2.get('0.0', 'end').strip())
        # create video objects
        for icam in camsList:
            videos[icam] = cv2.VideoCapture(fVideos[icam])
            if not videos[icam].isOpened(): 
                print("# Error opening video file %d" % icam)
            nFrames[icam] = round(videos[icam].get(cv2.CAP_PROP_FRAME_COUNT))
        nFrameMax = np.max(nFrames)
        # Allocate memory for bigTable 
        bigTable = np.ones((nFrameMax, 600), dtype=np.float32) * np.nan
        # Initialization before frame-by-frame loop 
        for icam in camsList:
            # load initial images
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
        # Data preparation before tracking
        iFrame = 0
        bigTable[iFrame, 0] = iFrame + 1 # for user, frame index is 1-based
        for icam in range(2):
            for iPoi in poisList:
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
        #    In demo 10, video 2, the 46th (iFrame=45) and 47th frame (iFrame=46)
        #    cannot be read, i.e., videos[icam].read() failed and went to 
        #    exception. The frame image remains the previous one (from a different camera).
        #    That resulted in the rest of ECC failed to track.
        lastReadFrame = [imgInits[0].copy(), imgInits[1].copy()]
        for iFrame in framesList:
            if iFrame <= 0: # this loop should starts from 1 as frame 0 has been done outside this loop
                continue
            if iFrame >= bigTable.shape[0]:
                break
            bigTable[iFrame, 0] = iFrame + 1
            for icam in camsList:
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
                for iPoi in poisList:
                    if iPoi >= nPoints[icam]:
                        continue
                    # define template
                    x0 = round(tmplts[icam][iPoi,2])
                    y0 = round(tmplts[icam][iPoi,3])
                    x1 = round(x0 + tmplts[icam][iPoi,4])
                    y1 = round(y0 + tmplts[icam][iPoi,5])
                    dx = tmplts[icam][iPoi, 0] - x0 # the difference between POI and tmplt up-left corner
                    dy = tmplts[icam][iPoi, 1] - y0 # the difference between POI and tmplt up-left corner
                    tmplt = imgInits[icam][y0:y1, x0:x1].copy()
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
                                print("# Warning: ECC got exception (Frame %d, Cam %d, Point %d)" % (iFrame+1, icam+1, iPoi+1))
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
                # end of loop of iPoi
            # end of loop of icam
            # print info
            if (iFrame+1) % 10 == 0:
                print("Frame %d completed ECC." % (iFrame+1) )
        # end of loop of iFrame
        # release videos
        for icam in range(2):
            if videos[icam].isOpened():
                videos[icam].release()
        # 
        # save bigTable to file
        event_btSaveData(event)
        pass
    # end of event_btImageTrack()
    # bind button "Image Track" to event_btImageTrack
    btImageTrack.bind("<ButtonRelease-1>", event_btImageTrack)

    # bind event of button "Save data" to event_btSaveData   
    def event_btSaveData(event):
        workDir = txWorkDir.get('0.0', 'end').strip()
        fBigTable = os.path.join(workDir, txSaveData.get('0.0', 'end').strip())
        np.savetxt(fBigTable, bigTable, delimiter=",")
    btSaveData.bind("<ButtonRelease-1>", event_btSaveData)

    # bind event of "Find best time lags"
    def event_btFindTimeLags(event):
        global bigTable, nFrames, cmats, dvecs, rvecs, tvecs
        global prjErr
        print("# event_btFindTimeLags().")
        # Range of possible lags. t_lag_trials needs to be continuous integer list.
        tLagTrialsText = txPossibleLag.get('0.0', 'end').strip()
        t_lag_trials = np.array(colonRangeToIntList(tLagTrialsText), dtype=int)
        t_lag_bounds = [np.min(t_lag_trials), np.max(t_lag_trials)]
        # Range of points for sync analysis
        poiRangeForSyncTxt = txSyncPoiRange.get('0.0', 'end').strip()
        poiRangeForSync = np.array(colonRangeToIntList(poiRangeForSyncTxt), dtype=int)
        # Range of sync analysis frames. t_interest_range needs to be continuous integer list.
        t_interest_text = txSyncFrameRange.get('0.0', 'end').strip()
        t_interest = np.array(colonRangeToIntList(t_interest_text), dtype=int)
        t_interest_range = [np.min(t_interest), np.max(t_interest)]
        # Clear Time lags (each point)
        txLags.delete('0.0', 'end')
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
        if t_interest_range[1] - t_lag_bounds[0] > (nFrames[1]-1):
            t_interest_range[1] = int(nFrames[1]-1 + t_lag_bounds[0]) 
            t_interest_range_forcedToChange = True
        if t_interest_range_forcedToChange:
            print("# Warning: t_lag_bounds is changed from (%d,%d) to (%d,%d)" %
                (t_interest_range_original[0], t_interest_range_original[1], 
                t_interest_range[0],          t_interest_range[1]))
        # t_interest[icam]: time tags of frames of interests 
        #   For example, if you are only interested in the triangulation of 
        #   frames 300 to 3900 in cam 0, t_interest = [300, 301, ..., 3900]
        #   and t_v2_interest = [330, 331, 332., ...] (if t_lag_trial is 30)
        nFrameInterest = round(t_interest_range[1] - 
                            t_interest_range[0])
        t_interest = np.linspace(t_interest_range[0], 
                                t_interest_range[1],
                                nFrameInterest)
        #
        imgPoints1_xyi_thisPoi_interest = np.zeros((t_interest.size, 2), dtype=np.float64)
        imgPoints2_xyi_thisPoi_interest = np.zeros((t_interest.size, 2), dtype=np.float64)
        prjErr = np.zeros((min(nPoints), t_lag_trials.size), dtype=float)
        ticMeasures = np.zeros(100)
        for ilag in range(t_lag_trials.size):
            tic = time.time()
            t_lag = t_lag_trials[ilag]
            # t_v1 and t_v2: based on the t_tag, set time tags of every frame   
            # t_v1[] would be [0., 1., 2., ...]
            # t_v2[] would be [30., 31., 32., ...] if video 2 lags 30 seconds (turned on 30s later)    
            t_v1 = np.linspace(0, nFrames[0] - 1, round(nFrames[0]))
            t_v2 = np.linspace(0, nFrames[1] - 1, round(nFrames[1])) + t_lag
            ticMeasures[0] += time.time() - tic
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
#                print("# If time lag is %.3fs, average projection error of POI %d is %.3f" % (t_lag, iPoi+1, err) )
            # end of for iPoi in range(min(nPoints))
            avgErr = np.sum(prjErr[:,ilag]) / prjErr.shape[0]
            print("# If time lag is %.3fs, average projection error of all points is %.3f" % (t_lag, avgErr) )
        # end of for ilag in range(t_lag_trials.size)
        #
        t_lags = np.zeros(min(nPoints), dtype=float)
        for iPoi in range(min(nPoints)):
            # find the t_lag which results in minimum prjErr[iPoi, :]
            ilagmin = np.argmin(prjErr[iPoi,:])
            t2 = t_lag_trials[ilagmin]
            t1 = t2 - 1
            t3 = t2 + 1
            e1 = prjErr[iPoi, ilagmin - 1]
            e2 = prjErr[iPoi, ilagmin]
            e3 = prjErr[iPoi, ilagmin + 1]
            # shift time axis so that t2_ = 0, t2_ = t2 - t2
            tmat = np.array([1, -1, 1, 0, 0, 1, 1, 1, 1], dtype=float).reshape(3,3)
            evec = np.array([e1, e2, e3], dtype=float).reshape(3, 1)
            coef = (np.linalg.inv(tmat) @ evec).flatten()
            t_lags[iPoi] = -.5 * coef[1] / coef[0] + t2
        # end of for iPoi in range(min(nPoints))
        # display t_lags to 
        txtmp = '\t'.join(map(str, t_lags))
        txLags.delete('0.0', 'end')
        txLags.insert('0.0', txtmp)
        pass
    # end of def event_btFindTimeLags(event):
    btFindTimeLags.bind("<ButtonRelease-1>", event_btFindTimeLags)

    # bind event of buttonn "Triangulation"
    def event_btTriang(event):
        global nFrames, bigTable
        # based on best time lags, do triangulation
        print('# event_btTriang')
        # get time lags (time lag of each point) from txLags
        tLagsText = txLags.get('0.0', 'end').strip()
        tLagsText = tLagsText.replace(',', ' ').replace('\t',' ').replace('\n',' ').replace('\r',' ').replace(';',' ')
        tLagsText = tLagsText.replace('(', ' ').replace(')',' ').replace('[',' ').replace(']',' ')
        t_lags = np.fromstring(tLagsText, sep=' ', dtype=float)
        # Range of triangulation frames. 
        t_interest_text = txTriangFrameRange.get('0.0', 'end').strip()
        t_interest = np.array(colonRangeToIntList(t_interest_text), dtype=int)
        t_interest -= 1 # convert to 0-based index
        # Range of triangulation points. 
        poisTriangulation_text = txTriangPoiRange.get('0.0', 'end').strip()
        poisTriangulation = np.array(colonRangeToIntList(poisTriangulation_text), dtype=int)
        poisTriangulation -= 1 # convert to 0-based index
        # triangulation is done point by point.
        imgPoints1_xyi_thisPoi_interest = np.zeros((t_interest.size, 2), dtype=np.float64)
        imgPoints2_xyi_thisPoi_interest = np.zeros((t_interest.size, 2), dtype=np.float64)
        # run loop (each point)
        for iPoi in poisTriangulation:
            # get time lag of this point, as each point has different time lag
            t_lag = t_lags[iPoi]
            t_v1 = np.linspace(0, nFrames[0] - 1, round(nFrames[0]))
            t_v2 = np.linspace(0, nFrames[1] - 1, round(nFrames[1])) + t_lag
            # image points of this point in cam 0
            icam = 0
            xi = bigTable[:, iPoi*5+200*icam+1]
            yi = bigTable[:, iPoi*5+200*icam+2]
            imgPoints1_xyi_thisPoi_interest[:,0] = xi[round(t_interest[0]):(round(t_interest[-1])+1)].copy()
            imgPoints1_xyi_thisPoi_interest[:,1] = yi[round(t_interest[0]):(round(t_interest[-1])+1)].copy()
            # image points of this point in cam 1
            # create interpolation objects for cam 2 (v2)
            icam = 1
            xi = bigTable[:, iPoi*5+200*icam+1]
            yi = bigTable[:, iPoi*5+200*icam+2]
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
            intpv2xi = scipy.interpolate.interp1d(t_v2, xi, kind='cubic')
            intpv2yi = scipy.interpolate.interp1d(t_v2, yi, kind='cubic')
            imgPoints2_xyi_thisPoi_interest[:,0] = intpv2xi(t_interest)
            imgPoints2_xyi_thisPoi_interest[:,1] = intpv2yi(t_interest)
            # Triangulation
            objPoints, objPoints1, objPoints2, \
                prjPoints1, prjPoints2, prjErrors1, prjErrors2 = \
                triangulatePoints2(cmats[0], dvecs[0], rvecs[0], tvecs[0], 
                                cmats[1], dvecs[1], rvecs[1], tvecs[1], 
                                imgPoints1_xyi_thisPoi_interest, 
                                imgPoints2_xyi_thisPoi_interest)
            # write to bigTable
            bigTable[t_interest, 401 + iPoi * 7] = objPoints[:,0]
            bigTable[t_interest, 402 + iPoi * 7] = objPoints[:,1]
            bigTable[t_interest, 403 + iPoi * 7] = objPoints[:,2]
            bigTable[t_interest, 404 + iPoi * 7] = prjErrors1[:,0]
            bigTable[t_interest, 405 + iPoi * 7] = prjErrors1[:,1]
            bigTable[t_interest, 406 + iPoi * 7] = prjErrors2[:,0]
            bigTable[t_interest, 407 + iPoi * 7] = prjErrors2[:,1]
        # end of for iPoi in range(min(nPoints))
        pass
    # end of def event_btTriang(event)
    btTriang.bind('<ButtonRelease-1>', event_btTriang)

    # bind event of "Save config"
    def event_btSaveConfig(event):
        workDir = txWorkDir.get("1.0", "end").strip()
        confFilebase = 'Stereosync_config.xml'
        confFilename = os.path.join(workDir, confFilebase)
        # Create a FileStorage object for writing the XML file
        fs = cv2.FileStorage(confFilename, cv2.FILE_STORAGE_WRITE_BASE64)
        # Save config to file
        #   working directory
        workDir = txWorkDir.get('1.0', 'end').strip()
        fs.write('working_directory', workDir)
        #   video file 1
        vidFile1 = txVidFile1.get('1.0', 'end').strip()
        fs.write('video_file_1', vidFile1)
        #   video file 2
        vidFile2 = txVidFile2.get('1.0', 'end').strip()
        fs.write('video_file_2', vidFile2)
        #   calibration file 1
        calibFile1 = txCalibFile1.get('1.0', 'end').strip()
        fs.write('calibration_file_1', calibFile1)
        #   calibration file 2
        calibFile2 = txCalibFile1.get('1.0', 'end').strip()
        fs.write('calibration_file_2', calibFile2)
        #   templates file 1
        tmpltFile1 = txTmpltFile1.get('1.0', 'end').strip()
        fs.write('templates_file_1', tmpltFile1)
        #   templates file 2
        tmpltFile2 = txTmpltFile2.get('1.0', 'end').strip()
        fs.write('templates_file_2', tmpltFile2)
        #   range of tracking cameras
        trackCams = txTrackCamRange.get('1.0', 'end').strip()
        fs.write('range_of_tracking_cameras', trackCams)
        #   range of tracking points
        trackPois = txTrackPointRange.get('1.0', 'end').strip()
        fs.write('range_of_tracking_points', trackPois)
        #   range of tracking frames
        trackFrames = txTrackFrameRange.get('1.0', 'end').strip()
        fs.write('range_of_tracking_frames', trackFrames)
        #   range of possible lags
        possibleLags = txPossibleLag.get('1.0', 'end').strip()
        fs.write('range_of_possible_lags', possibleLags)
        #   range of points for sync analysis
        syncPois = txSyncPoiRange.get('1.0', 'end').strip()
        fs.write('range_of_points_for_sync_analysis', syncPois)
        #   range of frames for sync analysis
        syncFrames = txSyncFrameRange.get('1.0', 'end').strip()
        fs.write('range_of_frames_for_sync_analysis', syncFrames)
        #   range of points for triangulation
        triangPois = txTriangPoiRange.get('1.0', 'end').strip()
        fs.write('range_of_points_for_triangulation', triangPois)
        #   range of frames for triangulation
        triangFrames = txTriangFrameRange.get('1.0', 'end').strip()
        fs.write('range_of_frames_for_triangulation', triangFrames)
        # release
        fs.release()
        print("# Config file saved.")
    #
    # Make <Tab> key switching to next widget, not adding tab in Text
    #
    def tabkey_focus_next(event):
        """Handles Tab key press and switches focus to the next text widget."""
        widget = event.widget  # Get the widget that triggered the event (the current text widget)
        widget.tk_focusNext().focus_set()  # Get the next widget in the focus order and set focus on it
        return "break"  # Prevent further default tab handling by Tkinter
    
    txWorkDir.bind("<Tab>", tabkey_focus_next)
    txVidFile1.bind("<Tab>", tabkey_focus_next)
    txVidFile2.bind("<Tab>", tabkey_focus_next)
    txCalibFile1.bind("<Tab>", tabkey_focus_next)
    txCalibFile2.bind("<Tab>", tabkey_focus_next)
    txTmpltFile1.bind("<Tab>", tabkey_focus_next)
    txTmpltFile2.bind("<Tab>", tabkey_focus_next)
    txTrackCamRange.bind("<Tab>", tabkey_focus_next)
    txTrackFrameRange.bind("<Tab>", tabkey_focus_next)
    txTrackPointRange.bind("<Tab>", tabkey_focus_next)
    txSyncFrameRange.bind("<Tab>", tabkey_focus_next)
    txSyncPoiRange.bind("<Tab>", tabkey_focus_next)
    txSaveConfig.bind("<Tab>", tabkey_focus_next)
    txSaveData.bind("<Tab>", tabkey_focus_next)
    txPossibleLag.bind("<Tab>", tabkey_focus_next)
    txTriangFrameRange.bind("<Tab>", tabkey_focus_next)
    txTriangPoiRange.bind("<Tab>", tabkey_focus_next)
    txLags.bind("<Tab>", tabkey_focus_next)

    # function of loading config
    def loadConfig(event=None):
        global fTmplts, tmplts, nPoints
        # Create a FileStorage object for reading the XML file
        workDir = txWorkDir.get("1.0", "end").strip()
        confFilebase = txLoadConfig.get('0.0', 'end').strip()
        confFilename = os.path.join(workDir, confFilebase)
        if os.path.exists(confFilename) == False:
            print("# Error: Cannot file config file: %s" % confFilename)
            return
        fs = cv2.FileStorage(confFilename, cv2.FILE_STORAGE_READ)
        # Load config to file
        #   video file 1
        vidFile1 = fs.getNode('video_file_1').string()
        txVidFile1.delete('1.0', 'end')
        txVidFile1.insert('1.0', vidFile1)
        #   video file 2
        vidFile2 = fs.getNode('video_file_2').string()
        txVidFile2.delete('1.0', 'end')
        txVidFile2.insert('1.0', vidFile2)
        #   calibration file 1
        calibFile1 = fs.getNode('calibration_file_1').string()
        txCalibFile1.delete('1.0', 'end')
        txCalibFile1.insert('1.0', calibFile1)
        #   calibration file 2
        calibFile2 = fs.getNode('calibration_file_2').string()
        txCalibFile2.delete('1.0', 'end')
        txCalibFile2.insert('1.0', calibFile2)
        #   templates file 1
        tmpltFile1 = fs.getNode('templates_file_1').string()
        txTmpltFile1.delete('1.0', 'end')
        txTmpltFile1.insert('1.0', tmpltFile1)
        #   templates file 2
        tmpltFile2 = fs.getNode('templates_file_2').string()
        txTmpltFile2.delete('1.0', 'end')
        txTmpltFile2.insert('1.0', tmpltFile2)

        #   range of tracking cameras
        trackCams = fs.getNode('range_of_tracking_cameras').string()
        txTrackCamRange.delete('1.0', 'end')
        txTrackCamRange.insert('1.0', trackCams)
        #   range of tracking points
        trackPois = fs.getNode('range_of_tracking_points').string()
        txTrackPointRange.delete('1.0', 'end')
        txTrackPointRange.insert('1.0', trackPois)
        #   range of tracking frames
        trackFrames = fs.getNode('range_of_tracking_frames').string()
        txTrackFrameRange.delete('1.0', 'end')
        txTrackFrameRange.insert('1.0', trackFrames)
        #   range of possible lags
        possibleLags = fs.getNode('range_of_possible_lags').string()
        txPossibleLag.delete('1.0', 'end')
        txPossibleLag.insert('1.0', possibleLags)
        #   range of points for sync analysis
        syncPois = fs.getNode('range_of_points_for_sync_analysis').string()
        txSyncPoiRange.delete('1.0', 'end')
        txSyncPoiRange.insert('1.0', syncPois)
        #   range of frames for sync analysis
        syncFrames = fs.getNode('range_of_frames_for_sync_analysis').string()
        txSyncFrameRange.delete('1.0', 'end')
        txSyncFrameRange.insert('1.0', syncFrames)
        #   range of points for triangulation
        triangPois = fs.getNode('range_of_points_for_triangulation').string()
        txTriangPoiRange.delete('1.0', 'end')
        txTriangPoiRange.insert('1.0', triangPois)
        #   range of frames for triangulation
        triangFrames = fs.getNode('range_of_frames_for_triangulation').string()
        txTriangFrameRange.delete('1.0', 'end')
        txTriangFrameRange.insert('1.0', triangFrames)
        # release
        fs.release()
        # set nFrames
        fVideos[0] = os.path.join(workDir, txVidFile1.get('0.0', 'end').strip())
        fVideos[1] = os.path.join(workDir, txVidFile2.get('0.0', 'end').strip())
        for icam in range(2):
            vx = cv2.VideoCapture(fVideos[icam])
            if vx.isOpened(): 
                nFrames[icam] = round(vx.get(cv2.CAP_PROP_FRAME_COUNT))
                print('# Video %d has %d frames.' % (icam+1, nFrames[icam]))
            else:
                print("# Error opening video file %d:%s" % (icam, fVideos[icam]))
        # load camera parameters: cmats, dvecs, rvecs, tvecs
        fCalibs=[None, None]
        fCalibs[0] = os.path.join(workDir, txCalibFile1.get('0.0', 'end').strip())
        fCalibs[1] = os.path.join(workDir, txCalibFile2.get('0.0', 'end').strip())
        for icam in range(2):
            # load calibration parameters
            try:
                camParam = np.loadtxt(fCalibs[icam], delimiter=',')
                rvecs[icam] = camParam[2:5]
                tvecs[icam] = camParam[5:8]
                cmats[icam] = camParam[8:17].reshape(3,3)
                dvecs[icam] = camParam[17:]
            except:
                print("# Failed to load camera calibration file %s" % fCalibs[icam])
        # load templates
        fTmplts = [None, None]
        nPoints = [0, 0]
        fTmplts[0] = os.path.join(workDir, txTmpltFile1.get('0.0', 'end').strip())
        fTmplts[1] = os.path.join(workDir, txTmpltFile2.get('0.0', 'end').strip())
        for icam in range(2):
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

#        nPoints = 
        print("# Config file loaded.")
    # end of def loadConfig(event=None):
        
    # bind event of btLoadConfig
    btLoadConfig.bind("<ButtonRelease-1>", loadConfig)

    # function of load data
    def loadData(event=None):
        global bigTable
        workDir = txWorkDir.get("1.0", "end").strip()
        dataFilebase = txLoadData.get('0.0', 'end').strip()
        dataFilename = os.path.join(workDir, dataFilebase)
        bigTable = np.loadtxt(dataFilename, delimiter=',')
        print("# Loaded data size: (%d, %d)" % (bigTable.shape[0], bigTable.shape[1]))
    # end of def loadData(event=None)

    # bind event of btLoadData
    btLoadData.bind('<ButtonRelease-1>', loadData)

    btSaveConfig.bind("<ButtonRelease-1>", event_btSaveConfig)
    
    win.mainloop()







if __name__ == '__main__':
    tkStereosync()
    
    