import os, sys
import time, datetime
import tkinter as tk
import tkinter.filedialog
import tkinter.messagebox
#import tkinter.scrolledtext
import numpy as np
import cv2 as cv

from improCalib import newArrayByMapping, countCalibPoints
from improMisc import uigetfile, uiputfile, uigetfiles_tupleFullpath
from improStrings import npFromString, stringFromNp, npFromTupleNp
from drawPoints import drawPoints
from imshow2 import imshow2
from writeCamera import writeCamera
from draw3dMesh import draw3dMesh
from chessboard_object_points import chessboard_object_points

def tkCalib_printMessage(msg: str):
    strfNow = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("%s: %s" % (strfNow, msg))
    return

def tkCalib():
#    camParams = np.zeros(14, dtype=np.float64)
    imgSize = np.zeros(2, dtype=int)
    rvec = np.zeros((3, 1), dtype=float)
    tvec = np.zeros((3, 1), dtype=float)
    cmat = np.eye(3, dtype=float)
    dvec = np.zeros((1, 5), dtype=float)
    # constants
    strFlags = ['CALIB_USE_INTRINSIC_GUESS', 'CALIB_FIX_ASPECT_RATIO',
        'CALIB_FIX_PRINCIPAL_POINT', 'CALIB_ZERO_TANGENT_DIST',
        'CALIB_FIX_FOCAL_LENGTH', 'CALIB_FIX_K1',
        'CALIB_FIX_K2', 'CALIB_FIX_K3', 'CALIB_FIX_K4',
        'CALIB_FIX_K5', 'CALIB_FIX_K6', 'CALIB_RATIONAL_MODEL',
        'CALIB_THIN_PRISM_MODEL', 'CALIB_FIX_S1_S2_S3_S4',
        'CALIB_TILTED_MODEL', 'CALIB_FIX_TAUX_TAUY', 'CALIB_USE_QR',
        'CALIB_FIX_TANGENT_DIST', 'CALIB_FIX_INTRINSIC',
        'CALIB_SAME_FOCAL_LENGTH', 'CALIB_ZERO_DISPARITY',
        'CALIB_USE_LU', 'CALIB_USE_EXTRINSIC_GUESS' ]
    #
    global win
    win = tk.Tk()
    win.geometry("1600x900")
    frame1 = tk.Frame(win)
    frame1.pack(side=tk.LEFT, anchor='n')
    frame2 = tk.Frame(win)
    frame2.pack(side=tk.LEFT, anchor='n')
    frame3 = tk.Frame(win)
    frame3.pack(side=tk.LEFT, anchor='n')
    frame4 = tk.Frame(win)
    frame4.pack(side=tk.LEFT, anchor='n')

    # #######################################################
    # a pair of ratio buttons of [1-image calib] and [chessboard calib]
    # #######################################################
    # define function of radio button
    # if the ratio button of [1-image calib] is clicked,
    # it enables 'Calibrate camera (1-photo)' button 
    # and disables 'Find chessboard corners' button and 'Calibrate camera (chessboard)' button
    # if the ratio button of [chessboard calib] is clicked,
    # it enables 'Find chessboard corners' button and 'Calibrate camera (chessboard)' button
    # and disables 'Calibrate camera (1-photo)' button
    # Note. The btCalib, btFindCbCorners, and btCalibCb are defined in the following sections
    #       after the radio buttons.
    def rb_clicked():
        if rbVar.get() == 0:
            btCalib.configure(state='normal')
            btFindCbCorners.configure(state='disabled')
            edCbParams.config(state=tk.DISABLED, bg="#F0F0F0")
            btCalibCb.configure(state='disabled')
            edGridImgIdx.configure(state='disabled')
            pass
        else:
            btCalib.configure(state='disabled')
            btFindCbCorners.configure(state='normal')
            edCbParams.config(state=tk.NORMAL, bg="white")
            btCalibCb.configure(state='normal')
            edGridImgIdx.configure(state='normal')
            pass
        return
    # create a pair of radio buttons
    rbVar = tk.IntVar()
    rb1 = tk.Radiobutton(frame1, text='single-image calib', variable=rbVar, value=0, command=rb_clicked)
    rb1.pack()
    rb2 = tk.Radiobutton(frame1, text='chessboard calib', variable=rbVar, value=1, command=rb_clicked)
    rb2.pack()
    # set initial value of radio button
    rbVar.set(0)

    # #######################################################
    # Button and edit text of [3D coordinates (World coord.)]
    # #######################################################
    # button of 3D coordinates (World coord.)
    btCoord3d = tk.Button(frame1, height=1, text='3D coordinates (World coord.)')
    btCoord3d.pack()
    # edit text of 3D coordinates
    edCoord3d = tk.Text(frame1, width=40, height=3, undo=True, 
                        autoseparators=True, maxundo=-1)
    edCoord3d.pack()
    #   set initial text for demonstration
    try:
        theMat = np.loadtxt(os.path.join(os.getcwd(), 'tkCalib_init_coord3d.npy'))
        theStr = stringFromNp(theMat, ftype='txtf', sep='\t')
        edCoord3d.delete(0., tk.END)
        edCoord3d.insert(0., theStr)
    except:
        tkCalib_printMessage("# Warning: Failed to load tkCalib_init_coord3d.npy")
        edCoord3d.delete(0., tk.END)
        edCoord3d.insert(0., ' 0 0 0 \n 1 0 0 \n 1 1 0 \n 0 1 0 \n nan nan nan')  
    # button command function 
    def btCoord3d_clicked():
        strPoints3d = edCoord3d.get(0., tk.END)
        try:
            points3d = npFromString(strPoints3d).reshape((-1, 3))
        except:
            tkCalib_printMessage('Cannot parse 3D points (size: %d)' % 
                                 (int(points3d.size)))
        strPointsImg= edCoordImg.get(0., tk.END)
        try:
            points2d = npFromString(strPointsImg).reshape((-1, 2))
        except:
            tkCalib_printMessage('Cannot parse 2D points (size: %d)' % 
                                 (int(points2d.size)))
        validsOfPoints3d, validsOfPoints2d, validCalibPoints, \
            idxAllToValid, idxValidToAll, validPoints3d, validPoints2d = \
            countCalibPoints(points3d, points2d)
        npts3d = points3d.shape[0]
        nval3d = np.sum(validsOfPoints3d)
        npts2d = points2d.shape[0]
        nval2d = np.sum(validsOfPoints2d)
        nValidCalibPoints = np.sum(validCalibPoints)
        # print info
        tkCalib_printMessage('Numbers of valid/total 3D points: %d/%d.' % 
                             (nval3d, npts3d))
        tkCalib_printMessage('Numbers of valid/total 2D points: %d/%d.' % 
                             (nval2d, npts2d))
        if (npts3d == npts2d):
            tkCalib_printMessage('Number of valid calibration points: %d' % 
                                 nValidCalibPoints)
            strValidPoints = '  Valid calib points (index is 0 based):'
            for i in range(npts3d):
                if validCalibPoints[i] == 1:
                    strValidPoints += ' %d' % i
            tkCalib_printMessage(strValidPoints)
        return
    # set button command function 
    btCoord3d.configure(command=btCoord3d_clicked)
    # #######################################################
    # Button and edit text of [Image 2D coordinates]
    # #######################################################
    # button of image coordinates
    btCoordImg = tk.Button(frame1, text='Image 2D coordinates')
    btCoordImg.pack()
    # edit text of image coordinates
    edCoordImg = tk.Text(frame1, width=40, height=3, undo=True, 
                         autoseparators=True, maxundo=-1)
    edCoordImg.pack()
    #   set initial text for demonstration
    edCoordImg.delete(0., tk.END)
    #   set initial text for demonstration
    try:
        theMat = np.loadtxt(os.path.join(os.getcwd(), 'tkCalib_init_coordImg.npy'))
        theStr = stringFromNp(theMat, sep='\t')
        edCoordImg.insert(0., theStr)
    except:
        tkCalib_printMessage("# Warning: Failed to load tkCalib_init_coordImg.npy")
        edCoordImg.insert(0., '1. 1. \n 2. 2. \nnan nan \n 1. 2. \n 3. 3.')
    # button command function
    def btCoordImg_clicked():
        strPoints3d = edCoord3d.get(0., tk.END)
        strPointsImg= edCoordImg.get(0., tk.END)
        try:
            points3d = npFromString(strPoints3d).reshape((-1, 3))
        except:
            tkCalib_printMessage('# Error: Cannot parse 3D points (size: %d)' % 
                                 (int(points3d.size)))
        try:
            points2d = npFromString(strPointsImg).reshape((-1, 2))
        except:
            tkCalib_printMessage('# Error: Cannot parse 2D points (size: %d)' % 
                                 (int(points2d.size)))
        validsOfPoints3d, validsOfPoints2d, validCalibPoints, \
            idxAllToValid, idxValidToAll, validPoints3d, validPoints2d = \
            countCalibPoints(points3d, points2d)
        npts3d = points3d.shape[0]
        nval3d = np.sum(validsOfPoints3d)
        npts2d = points2d.shape[0]
        nval2d = np.sum(validsOfPoints2d)
        nValidCalibPoints = np.sum(validCalibPoints)
        # print info
        tkCalib_printMessage('Numbers of valid/total 3D points: %d/%d.' % 
                             (nval3d, npts3d))
        tkCalib_printMessage('Numbers of valid/total 2D points: %d/%d.' % 
                             (nval2d, npts2d))
        if (npts3d == npts2d):
            tkCalib_printMessage('Number of valid calibration points: %d' % 
                                 nValidCalibPoints)
            strValidPoints = '  Valid calib points (index is 0 based):'
            for i in range(npts3d):
                if validCalibPoints[i] == 1:
                    strValidPoints += ' %d' % i
            tkCalib_printMessage(strValidPoints)
        return
    # set command     
    btCoordImg.configure(command=btCoordImg_clicked)  
    # #######################################################
    # Button of [Select calibration image(s) ...]
    # #######################################################
    btFile = tk.Button(frame1, text='Select calibration image(s) ...')
    btFile.pack()
    edFile = tk.Text(frame1, width=40, height=2)
    edFile.pack()
    #   set initial text for demonstration
    try:
        tfile = open(os.path.join(os.getcwd(), 'tkCalib_init_imgFile.txt'), "r")
        fname = tfile.read()
        tfile.close()
        edFile.delete(0., tk.END)
        edFile.insert(0., fname)
    except:
        edFile.insert(0., 'c:/images/calibration.bmp')
        
    #   btFile 'Select image ...'
    def btFile_clicked():
        # select file
        fpath = edFile.get(0., tk.END)
        try:
            # fpath could be 1 file or multiple files
            fpath = fpath.split()[0]
        except:
            pass
        initDir = os.path.split(fpath)[0]
        # get file(s) from file dialog (format: tuple of full paths)
        uFullpaths = uigetfiles_tupleFullpath(initialDirectory=initDir)
        # set edFile text to the full path of select file(s)
        # if user selects 1 file, edFile text would be the full path of the file
        # if user selects multiple files, edFile text would be full paths of all files and are separated with \n
        # the image size would be set to that of the first file (index [0])
        if len(uFullpaths) > 1:
            edFile.delete(0., tk.END)
            edFile.insert(0., '\n'.join(uFullpaths))
        else:
            edFile.delete(0., tk.END)
            edFile.insert(0., uFullpaths[0])
        btImgSize_clicked() 
        return
    # set button command function 
    btFile.configure(command=btFile_clicked)
            
    # #######################################################
    # Button of [Image size:] (width height)
    # #######################################################
    btImgSize = tk.Button(frame1, text='Image size: width height')
    btImgSize.pack()
    edImgSize = tk.Text(frame1, width=40, height=1)
    edImgSize.pack()
    #   set initial text for demonstration
    try:
        theMat = np.loadtxt(os.path.join(os.getcwd(), 'tkCalib_init_imgSize.npy')).astype(int)
        theStr = stringFromNp(theMat, sep='\t')
        edImgSize.delete(0., tk.END)
        edImgSize.insert(0., theStr)
    except:
        tkCalib_printMessage("# Warning: Cannot parse tkCalib_init_imgSize.npy")
        edImgSize.delete(0., tk.END)
        edImgSize.insert(0., '1920 1080')
    # define functions
    def btImgSize_clicked():
        # try to open the image file and check the image size
        try:
            fname = edFile.get(0., tk.END)
            # if fname contains multiple files with delimiter of \n \t or space,
            # we only take the first one.
            fname = fname.split()[0]
            img = cv.imread(fname)
            if type(img) == np.ndarray and img.shape[0] >= 1:
                imgH = img.shape[0]
                imgW = img.shape[1]
                edImgSize.delete(0., tk.END)
                edImgSize.insert(0., '%d %d' % (imgW, imgH))
            # update initial guess of cmat
            theStr = edCmatGuess.get(0., tk.END)
            theMat = npFromString(theStr).reshape((3,3))
            if theMat.shape[0] == 3 and theMat.shape[1] == 3:
                theMat[0, 2] = imgW / 2.0 - 0.5
                theMat[1, 2] = imgH / 2.0 - 0.5
                theStr = stringFromNp(theMat, sep='\t')
                edCmatGuess.delete(0., tk.END)
                edCmatGuess.insert(0., theStr)
        except:
            tk.messagebox.showerror(title="Error", message="Cannot read the image file.")
            print("Cannot read the image file.")
        strImgSize = edImgSize.get(0., tk.END)
        print('Image size: ', strImgSize)
        return
    # imgSizeFromBt() determines image size. 
    # imgSizeFromBt() is called by btCalib_clicked(), btDrawPoints_clicked(), 
    #   btDrawPointsUndistort_clicked(), 
    def imgSizeFromBt():
        strImgSize = edImgSize.get(0., tk.END)
        imgSize = npFromString(strImgSize)
        if type(imgSize) != np.ndarray or imgSize.size != 2:
            errMsg = '# error: Image size is invalid (edit text is %s)' % (strImgSize)
            tk.messagebox.showerror(title="Error", message="errMsg")
            return None
        return imgSize.astype(int).flatten()
    # set button command function
    btImgSize.configure(command=btImgSize_clicked)
    
    # #######################################################
    # Button of [Camera mat (init guess)]
    # #######################################################
    btCmatGuess = tk.Button(frame1, text='Camera mat (init guess)')
    btCmatGuess.pack()
    edCmatGuess = tk.Text(frame1, width=40, height=2, undo=True, 
                          autoseparators=True, maxundo=-1)
    edCmatGuess.pack()
    #   set initial text for demonstration
    try:
        theMat = np.loadtxt(os.path.join(os.getcwd(), 'tkCalib_init_cmatGuess.npy'))
        theStr = stringFromNp(theMat, sep='\t')
        edCmatGuess.delete(0., tk.END)
        edCmatGuess.insert(0., theStr)
    except:
        print("Warning: Cannot load tkCalib_init_cmatGuess.npy")
        edCmatGuess.delete(0., tk.END)
#        edCmatGuess.insert(0., ' 5000. 0 0 \n 0 5000. 0 \n 1999.5 1999.5 1')  # This is wrong. Incorrectly written in transposed way.
        edCmatGuess.insert(0., ' 5000. 0 1999.5 \n 0 5000. 1999.5 \n 0 0 1')  # revised on 2024/08/28
    #   button 'Camera mat (init guess)' as a label
    # defind button command function 
    def btCmatGuess_clicked():
        try:
            strCmatGuess = edCmatGuess.get(0., tk.END)
            cmatGuess = npFromString(strCmatGuess).reshape((3,3))
        except:
            cmatGuess = np.array([])
        print('Initial guess of camera matrix:\n', cmatGuess)
        return cmatGuess
    # set button command function 
    btCmatGuess.configure(command=btCmatGuess_clicked)
    # #######################################################
    # Button of [Distortion coeff. (init guess)]
    # #######################################################
    btDvecGuess = tk.Button(frame1, text='Distortion coeff. (init guess)')
    btDvecGuess.pack()
    edDvecGuess = tk.Text(frame1, width=40, height=2, undo=True, 
                          autoseparators=True, maxundo=-1)
    edDvecGuess.pack()
    #   set initial text for demonstration
    try:
        theMat = np.loadtxt(os.path.join(os.getcwd(), 'tkCalib_init_dvecGuess.npy'))
        theStr = stringFromNp(theMat, sep='\t')
        edDvecGuess.delete(0., tk.END)
        edDvecGuess.insert(0., theStr)
    except:
        print("Warning: Cannot load tkCalib_init_dvecGuess.npy")
        edDvecGuess.delete(0., tk.END)
        edDvecGuess.insert(0., '0. 0. 0. 0.')
    # define button command function 
    def btDvecGuess_clicked():
        strDvecGuess = edDvecGuess.get(0., tk.END)
        try:
            cdvecGuess = npFromString(strDvecGuess).reshape((1,-1))
        except:
            cdvecGuess = np.array([])
        print('Initial guess of distortion vector:\n', cdvecGuess)
        return cdvecGuess
    # set button command function 
    btDvecGuess.configure(command=btDvecGuess_clicked)

    # #######################################################
    # Button of [Find chessboard corners]
    # #######################################################
    btFindCbCorners = tk.Button(frame1, text='Find chessboard corners')
    btFindCbCorners.pack()
    edCbParams = tk.Text(frame1, width=40, height=1, undo=True, 
                          autoseparators=True, maxundo=-1)
    edCbParams.pack()
    #   set initial text for demonstration
    try:
        theMat = np.loadtxt(os.path.join(os.getcwd(), 'tkCalib_init_cboardParams.npy'))
#        theStr = stringFromNp(theMat, sep='\t')
        theStr = '%d %d %f %f' % (theMat[0], theMat[1], theMat[2], theMat[3])
        edCbParams.delete(0., tk.END)
        edCbParams.insert(0., theStr)
    except:
        print("Warning: Cannot load tkCalib_init_cboardParams.npy")
        edCbParams.delete(0., tk.END)
        edCbParams.insert(0., '7 7 25.4 25.4 ')
    # define button command function
    def btFindCbCorners_clicked():
        # get chessboard files
        try:
            cbFiles = edFile.get(0., tk.END).split()
            nCbFiles = len(cbFiles)
        except:
            tk.messagebox.showerror(title="Error", message="# Error: Cannot get calibration files.")
            return 
        # get chessboard parameters (nCornersX, nCornersY, dxCorners, dyCorners)
        try:
            strCbParams = edCbParams.get(0., tk.END)
            npCbParams = npFromString(strCbParams)
            nCornersX = int(npCbParams[0])
            nCornersY = int(npCbParams[1])
            dxCorners = float(npCbParams[2])
            dyCorners = float(npCbParams[3])
        except:
            tk.messagebox.showerror(title="Error", message="# Error: Cannot get chessboard information.")
            return 
        # calibration flags
        try:
#            calibFlags = int(edFlags.get())
            calibFlags = ck_clicked()
        except:
            tk.messagebox.showerror(title="Error", message="# Error: Cannot get flags.")
            return 
        # allocate arrays (?)
        nFiles = len(cbFiles)
        # generate object points of chessboard corners
        corners_object_points_one_picture = chessboard_object_points(nCornersX, nCornersY, dxCorners, dyCorners)
        # clone the object points for all images. Stack vertically, make it 3D, i.e., nCbFiles * (nCornersX * nCornersY) * 3
        corners_object_points = np.tile(corners_object_points_one_picture, (nCbFiles, 1, 1))
        # print it to text box
        theStr = stringFromNp(corners_object_points, ftype='txtf', sep='\t')
        edCoord3d.delete(0., tk.END)
        edCoord3d.insert(0., theStr)
        # try to find corners from the images
        try:
            imgPointsList = []
            cornerFoundPhotoIndices = []
            for icb in range(nFiles):
                fname = cbFiles[icb]
                img = cv.imread(fname, cv.IMREAD_GRAYSCALE)
                flags=cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE
                # The variable "found" is a boolean variable that indicates whether the corners are found or not.
                # If "found", the variable ptsThis will be a float32 (nCornersX * nCornersY) * 1 * 2 array (3D).
                # (If not "found", the variable ptsThis will be a None.)
                found, ptsThis = cv.findChessboardCorners(img, (nCornersX, nCornersY), flags)
                if found:
                    print('Found corners in %s' % (os.path.basename(fname)))
                    imgPointsList.append(ptsThis)
                    cornerFoundPhotoIndices.append(icb)
                else:
                    errMsg = '# Failed to find corners in %s' % (os.path.basename(fname))
                    tkCalib_printMessage(errMsg)
                    tk.messagebox.showerror(title='Warning', message=errMsg)
        except:
            errMsg = '# Failed to find corners'
            tk.messagebox.showerror(title='Error', message=errMsg)
            return
        if len(imgPointsList) <= 0:
            errMsg = '# Failed to find corners for all images.'
            tk.messagebox.showerror(title='Error', message=errMsg)
            return
        # 
        nFound = len(imgPointsList)
        # change the edFile text to the files that corners are found
        theStr = '\n'.join([cbFiles[i] for i in cornerFoundPhotoIndices])
        edFile.delete(0., tk.END)
        edFile.insert(0., theStr)
        # convert imgPointsList to imgPoints2f
        imgPoints2f = np.array(imgPointsList).reshape(nFound, -1, 2)
        # print detected corners to 
        theStr = stringFromNp(imgPoints2f, sep='\t')
        edCoordImg.delete(0., tk.END)
        edCoordImg.insert(0., theStr)
        print(cbFiles, nCbFiles)
        strCbParams = edCbParams.get(0., tk.END)
        pass
    # set button command function 
    btFindCbCorners.configure(command=btFindCbCorners_clicked)

    # #######################################################
    # Frame 2
    # #######################################################
    # Checkbuttons of calibration flags
    # #######################################################
    # Checkbuttons of calibration flags 
    #   allowing user to switch on/off of every flag
    #   once clicked, sum of flags is displaced 
    # init of flags 
    edFlags = tk.Entry(frame2)
    edFlags.pack(anchor=tk.W)
    edFlags.delete(0, tk.END)
    edFlags.insert(0, 'Calib flags: 0')
    edFlags.config(state= "disabled")
    ckFlags = []
    ckValues = []
    # checkbox function
    def ck_clicked():
        # calculate sum of flags
        flags = 0
        for i in range(len(strFlags)):
            if ckValues[i].get() == 1:
                flags += eval('cv.' + strFlags[i])
        # display sum of flags
        #   set edFlags text to Calib flags: %d
        edFlags.config(state= "normal")
        edFlags.delete(0, tk.END)
        edFlags.insert(0, 'Calib flags: %d' % (flags))
        edFlags.config(state= "disabled")
        return flags
    for i in range(len(strFlags)):
        # generate statement string for creating checkbutton
        #   E.g., for i == 0
        #   "tk.Checkbutton(win, text='CALIB_USE_INTRINSIC_GUESS (0)',
        #                   command=ck_clicked, variable=ckValues[i])"
        evalStr = "tk.Checkbutton(frame2, text='"
        evalStr += strFlags[i] + " (%d)" % (eval('cv.' + strFlags[i])) + "', "
        evalStr += "command=ck_clicked, "
        evalStr += "variable=ckValues[%d]" % (i) + ")"
        # create a tk.IntVar() for checkbutton value
        ckValues.append(tk.IntVar())
        # create a checkbutton
        ckFlags.append(eval(evalStr)) # 
        # position a checkbutton
        ckFlags[i].pack(anchor=tk.W)
    # disable some flag checks that is not supported in tkCalib
    # as tkCalib assumes the distortion coefficients are k1, k2, p1, p2, ...
    # rather than s1, s2, .... 
    ckFlags[12].config(state= "disabled")
    ckFlags[13].config(state= "disabled")
    ckFlags[14].config(state= "disabled")
    ckFlags[15].config(state= "disabled")
    ckFlags[16].config(state= "disabled")
#    ckFlags[17].config(state= "disabled")
#    ckFlags[18].config(state= "disabled")
#    ckFlags[19].config(state= "disabled")
#    ckFlags[20].config(state= "disabled")
    ckFlags[21].config(state= "disabled")
    ckFlags[22].config(state= "disabled")

    # set initial flags (edFlags)
    try:
        theMat = np.loadtxt(os.path.join(os.getcwd(), 'tkCalib_init_calibFlags.npy'))
        theInt = int(theMat)
        theStr = '%d' % theInt
        edFlags.delete(0, tk.END)
        edFlags.insert(0, theStr)
        for i in range(len(strFlags)):
            flagInt = eval('cv.' + strFlags[i])
            if theInt & flagInt > 0:
                ckValues[i].set(1)
            else:
                ckValues[i].set(0)
    except:
        tkCalib_printMessage('Cannot load calibration flags from file. Set default flags.')
        ckValues[0].set(1)  # to use intrinsic guess
        ckValues[2].set(1)  # to fix principal point
        ckValues[3].set(1)  # to zero tangent dist
        ckValues[6].set(1)  # to fix k2
        ckValues[7].set(1)  # to fix k3
        ckValues[8].set(1)  # to fix k4
        ckValues[9].set(1)  # to fix k5
        ckValues[10].set(1) # to fix k6
    ck_clicked()

    # #######################################################
    # Frame 3
    # #######################################################
    # Button of [Calibrate camera (1-photo)]
    # #######################################################
    btCalib = tk.Button(frame3, text='Calibrate camera (1-photo)', width=40, height=2)
    btCalib.pack()
    # define button command function 
    def btCalib_clicked():
        print("# *** Button [Calibrate camera] clicked:\\")
        # get object points and image points as Numpy array format
        # from text boxes (edCoord3d and edCoordImg)
        strPoints3d = edCoord3d.get(0., tk.END)
        strPoints2d = edCoordImg.get(0., tk.END)
        try:
            points3d = npFromString(strPoints3d).reshape((-1, 3))
        except:
            tkCalib_printMessage('Cannot parse 3D points (size: %d)' % 
                                 (int(points3d.size)))
        try:
            points2d = npFromString(strPoints2d).reshape((-1, 2))
        except:
            tkCalib_printMessage('Cannot parse 2D points (size: %d)' % 
                                 (int(points2d.size)))
        # remove nan points, get intersection set of non-nan points
        #　get the mapping
        validsOfPoints3d, validsOfPoints2d, validCalibPoints, \
            idxAllToValid, idxValidToAll, validPoints3d, validPoints2d = \
            countCalibPoints(points3d, points2d)
        # variables and parameters to run camera calibration 
        nCalPoints = validPoints3d.shape[0]
        objPoints = validPoints3d.reshape((1, nCalPoints, 3)).astype(np.float32)
        imgPoints = validPoints2d.reshape((1, nCalPoints, 2)).astype(np.float32)
        imgSize = imgSizeFromBt().astype(int).reshape(-1)
        cmat = npFromString(edCmatGuess.get(0., tk.END)).reshape((3,3))
        dvec = npFromString(edDvecGuess.get(0., tk.END)).reshape((1,-1))
        flags = ck_clicked()
        rvecs = np.array([])
        tvecs = np.array([])
        print("Calibrating camera based on %d 3D points." % objPoints.shape[1])
        print("Calibrating camera based on %d img points." % imgPoints.shape[1])
        # run calibration 
        ret, cmat, dvec, rvecs, tvecs = cv.calibrateCamera(
            objPoints, imgPoints, imgSize, cmat, dvec, rvecs, tvecs, flags) 
        # reshape the results (rvecs and tvecs to 3d arrays)
#        rvecs = npFromTupleNp(rvecs[0]).reshape((3,1))
#        tvecs = npFromTupleNp(tvecs[0]).reshape((3,1))
        # display result to text boxes
        edRvecs.delete(0., tk.END)
        edRvecs.insert(0., stringFromNp(rvecs[0].reshape(-1,1), sep='\t'))
        edTvecs.delete(0., tk.END)
        edTvecs.insert(0., stringFromNp(tvecs[0].reshape(-1,1), sep='\t'))
        edCmat.delete(0., tk.END)
        edCmat.insert(0., stringFromNp(cmat.reshape(3,3), sep='\t'))
        edDvec.delete(0., tk.END)
        edDvec.insert(0., stringFromNp(dvec.reshape(-1,1), sep='\t'))
        # calculate camera position
        #   rvecs and tvecs would be a tuple of 3x1 float array(s)
#        campos = tvecs.copy() # make campos the same type and shape with tvecs
        campos = np.zeros((3, 1), dtype=float)
#        for i in range(tvecs.shape[0]):
        r3 = cv.Rodrigues(rvecs[0])[0]
        r4 = np.eye(4, dtype=float)
        r4[0:3,0:3] = r3.copy()
        r4[0:3,3] = tvecs[0].reshape((3, 1))[:, 0]
        r4inv = np.linalg.inv(r4)
        campos = r4inv[0:3,3].reshape((3, 1))
        edCampos.delete(0., tk.END)
        edCampos.insert(0., stringFromNp(campos, sep='\t'))
        # project points 
        prjPointsValid, jacobian = cv.projectPoints(objPoints.reshape((-1,3)), 
                                                    rvecs[0], tvecs[0], cmat, dvec)
        prjPointsValid = prjPointsValid.reshape((-1,2))
        prjPointsAll, jacobian = cv.projectPoints(points3d.reshape((-1,3)), 
                                                    rvecs[0], tvecs[0], cmat, dvec)
        prjPointsAll = prjPointsAll.reshape((-1,2))
        # calculate projection errors
        imgPointsValid = imgPoints.reshape((-1, 2))
        prjErrorsValid = (prjPointsValid - imgPointsValid)
        # display
#        prjPointsAll = newArrayByMapping(prjPointsValid, idxAllToValid)
        edCoordPrj.delete(0., tk.END)
        edCoordPrj.insert(0., stringFromNp(prjPointsAll, sep='\t'))
        prjErrorsAll = newArrayByMapping(prjErrorsValid, idxAllToValid)
        edPrjErrors.delete(0., tk.END)
        edPrjErrors.insert(0., stringFromNp(prjErrorsAll, sep='\t'))
        # all-in-one-column:rvec/tvec/cmat9/dvec14/empty/campos
        vec31 = np.zeros((31, 1), dtype=float)
        vec31[0:2, 0] = imgSize[0:2]
        vec31[2:5, 0] = rvecs[0].reshape((-1, 1))[0:3, 0]
        vec31[5:8, 0] = tvecs[0].reshape((-1, 1))[0:3, 0]
        vec31[8:17, 0] = cmat.reshape((9, 1))[0:9, 0]
        lenDvec = dvec.size
        vec31[17:17+lenDvec, 0] = dvec.reshape((lenDvec, 1))[0:lenDvec, 0]
        strOneCol = stringFromNp(vec31, sep='\t') + "\n\n"
        strOneCol += stringFromNp(campos.reshape((-1, 1)), sep='\t')
        edOneCol.delete(0., tk.END)
        edOneCol.insert(0., strOneCol)
        return
    # set button command function 
    btCalib.configure(command=btCalib_clicked)

    # #######################################################
    # Button of [Calibrate camera (chessboard)]
    # #######################################################
    btCalibCb = tk.Button(frame3, text='Calibrate camera (chessboard)', width=40, height=2)
    btCalibCb.pack()
    # define button command function 
    def btCalibCb_clicked():
        print("# *** Button [Calibrate camera (chessboard)] clicked:\\")
        # get object points and image points as Numpy array format
        # from text boxes (edCoord3d and edCoordImg)
        strPoints3d = edCoord3d.get(0., tk.END)
        strPoints2d = edCoordImg.get(0., tk.END)
        try:
            points3d = npFromString(strPoints3d).reshape((-1, 3))
        except:
            tkCalib_printMessage('Cannot parse 3D points (size: %d)' % 
                                 (int(points3d.size)))
        try:
            points2d = npFromString(strPoints2d).reshape((-1, 2))
        except:
            tkCalib_printMessage('Cannot parse 2D points (size: %d)' % 
                                 (int(points2d.size)))
            return

        # get number of chessboard files 
        try:
            cbFiles = edFile.get(0., tk.END).split()
            nCbFiles = len(cbFiles)
        except:
            tk.messagebox.showerror(title="Error", message="# Error: Cannot get calibration files.")
            return 

        # get chessboard parameters (nCornersX, nCornersY, dxCorners, dyCorners)
        try:
            strCbParams = edCbParams.get(0., tk.END)
            npCbParams = npFromString(strCbParams)
            nCornersX = int(npCbParams[0])
            nCornersY = int(npCbParams[1])
#            dxCorners = float(npCbParams[2])
#            dyCorners = float(npCbParams[3])
            nCbPhotos = points2d.shape[0] // (nCornersX * nCornersY)
        except:
            tk.messagebox.showerror(title="Error", message="# Error: Cannot get chessboard information.")
            return 
        # remove nan points, get intersection set of non-nan points
        #　get the mapping
        validsOfPoints3d, validsOfPoints2d, validCalibPoints, \
            idxAllToValid, idxValidToAll, validPoints3d, validPoints2d = \
            countCalibPoints(points3d, points2d)
        # check number of points 
        if validPoints3d.shape[0] != nCbPhotos * nCornersX * nCornersY:
            tkCalib_printMessage('Error: Number of calibration points (3D) (%d) does not match photos (%d) * corners (%dx%d)' % 
                                    (validPoints3d.shape[0], nCbPhotos, nCornersX, nCornersY))
            tkCalib_printMessage('       Calibration cancelled.')
            return
        if validPoints2d.shape[0] != nCbPhotos * nCornersX * nCornersY:
            tkCalib_printMessage('Error: Number of calibration points (image) (%d) does not match photos (%d) * corners (%dx%d)' % 
                                    (validPoints3d.shape[0], nCbPhotos, nCornersX, nCornersY))
            tkCalib_printMessage('       Calibration cancelled.')
            return
        # variables and parameters to run camera calibration 
        objPoints = validPoints3d.reshape((nCbPhotos, nCornersX * nCornersY, 3)).astype(np.float32)
        imgPoints = validPoints2d.reshape((nCbPhotos, nCornersX * nCornersY, 2)).astype(np.float32)
        imgSize = imgSizeFromBt().astype(int).reshape(-1)
        cmat = npFromString(edCmatGuess.get(0., tk.END)).reshape((3,3))
        dvec = npFromString(edDvecGuess.get(0., tk.END)).reshape((1,-1))
        flags = ck_clicked()
        rvecs = np.array([])
        tvecs = np.array([])
        print("Calibrating camera based on %d chessboard photos %dx%d corners" % (nCbPhotos, nCornersX, nCornersY))
        # run calibration 
        ret, cmat, dvec, rvecs, tvecs = cv.calibrateCamera(
            objPoints, imgPoints, imgSize, cmat, dvec, rvecs, tvecs, flags) 
        # reshape the results (rvecs and tvecs to 3d arrays)
#        rvecs = npFromTupleNp(rvecs[0]).reshape((3,1))
#        tvecs = npFromTupleNp(tvecs[0]).reshape((3,1))
        # display result to text boxes
        edRvecs.delete(0., tk.END)
        edRvecs.insert(0., stringFromNp(np.array(rvecs).reshape(-1,3), sep='\t'))
        edTvecs.delete(0., tk.END)
        edTvecs.insert(0., stringFromNp(np.array(tvecs).reshape(-1,3), sep='\t'))
        edCmat.delete(0., tk.END)
        edCmat.insert(0., stringFromNp(cmat.reshape(3,3), sep='\t'))
        edDvec.delete(0., tk.END)
        edDvec.insert(0., stringFromNp(dvec.reshape(-1,1), sep='\t'))
        # clear edCampos, edCoordPrj, edPrjErrors, edOneCol
        # edCampos: nCbPhotos * 3
        # edCoordPrj: (nCbPhotos * nCornersX * nCornersY) * 2
        # edPrjErrors: (nCbPhotos * nCornersX * nCornersY) * 2
        # edOneCol: nCbPhotos * _____
        # calculate camera position
        #   rvecs and tvecs would be a tuple of 3x1 float array(s)
#        campos = tvecs.copy() # make campos the same type and shape with tvecs
        for icam in range(nCbPhotos):
            campos = np.zeros((3, 1), dtype=float)
            r3 = cv.Rodrigues(rvecs[icam])[0]
            r4 = np.eye(4, dtype=float)
            r4[0:3,0:3] = r3.copy()
            r4[0:3,3] = tvecs[icam].reshape((3, 1))[:, 0]
            r4inv = np.linalg.inv(r4)
            campos = r4inv[0:3,3].reshape((3, 1))
            edCampos.insert(tk.END, stringFromNp(campos.flatten(), sep='\t')+'\n')
            # project points 
            prjPointsValid, jacobian = cv.projectPoints(
                objPoints.reshape(-1, (nCornersX*nCornersY), 3)[icam].reshape((-1,3)),
                rvecs[icam], tvecs[icam], cmat, dvec)
            prjPointsValid = prjPointsValid.reshape((-1,2))
            edCoordPrj.insert(tk.END, stringFromNp(prjPointsValid, sep='\t')+'\n')
            # calculate projection errors
            imgPointsValid = imgPoints.reshape(-1, (nCornersX*nCornersY), 2)[icam].reshape((-1,2))
            prjErrorsValid = (prjPointsValid - imgPointsValid)
            edPrjErrors.insert(tk.END, stringFromNp(prjErrorsValid, sep='\t')+ '\n')
            # all-in-one-column:img_W/img_H/rvec/tvec/cmat9/dvec14
            vec31 = np.zeros((31, 1), dtype=float)
            vec31[0:2, 0] = imgSize[0:2]
            vec31[2:5, 0] = rvecs[icam].reshape((-1, 1))[0:3, 0]
            vec31[5:8, 0] = tvecs[icam].reshape((-1, 1))[0:3, 0]
            vec31[8:17, 0] = cmat.reshape((9, 1))[0:9, 0]
            lenDvec = dvec.size
            vec31[17:17+lenDvec, 0] = dvec.reshape((lenDvec, 1))[0:lenDvec, 0]
            strOneCol = stringFromNp(vec31.flatten(), sep='\t') + "\n\n"
            edOneCol.insert(tk.END, strOneCol+'\n')
        return    
    # set button command function 
    btCalibCb.configure(command=btCalibCb_clicked)

    # #######################################################
    # Button of [Draw projection image]
    # #######################################################
    btDrawPoints = tk.Button(frame3, text='Draw points and grid', width=40, height=1)
    btDrawPoints.pack()
    # #######################################################
    # Button of [Draw projection image on an undistorted image]
    # #######################################################
    btDrawPointsUndist = tk.Button(frame3, text='Draw points and grid (undistorted)', 
                                   width=40, height=1)
    btDrawPointsUndist.pack()
    # #######################################################
    # Text of grid coordinates (Xs, Ys, and Zs)
    # #######################################################
    lbGridXs = tk.Label(frame3, text='xs of grid')
    lbGridXs.pack()
    edGridXs = tk.Entry(frame3, width=40)
    edGridXs.pack()
    lbGridYs = tk.Label(frame3, text='ys of grid')
    lbGridYs.pack()
    edGridYs = tk.Entry(frame3, width=40)
    edGridYs.pack()
    lbGridZs = tk.Label(frame3, text='zs of grid')
    lbGridZs.pack()
    edGridZs = tk.Entry(frame3, width=40)
    edGridZs.pack()
    lbGridImgIdx = tk.Label(frame3, text='Index of chessboard photos (1-based index)')
    lbGridImgIdx.pack()
    edGridImgIdx = tk.Entry(frame3, width=10)
    edGridImgIdx.insert(0, '1')
    edGridImgIdx.pack()

    #   set initial text for demonstration
    try:
        theMat = np.loadtxt(os.path.join(os.getcwd(), 'tkCalib_init_gridXs.npy'))
        theStr = stringFromNp(theMat, sep='\t')
        edGridXs.delete(0, tk.END)
        edGridXs.insert(0, theStr)
        theMat = np.loadtxt(os.path.join(os.getcwd(), 'tkCalib_init_gridYs.npy'))
        theStr = stringFromNp(theMat, sep='\t')
        edGridYs.delete(0, tk.END)
        edGridYs.insert(0, theStr)
        theMat = np.loadtxt(os.path.join(os.getcwd(), 'tkCalib_init_gridZs.npy'))
        theStr = stringFromNp(theMat, sep='\t')
        edGridZs.delete(0, tk.END)
        edGridZs.insert(0, theStr)
    except:
        tkCalib_printMessage("# Warning: Cannot parse tkCalib_init_gridXs/Ys/Zs.npy")
        edGridXs.delete(0, tk.END)
        edGridYs.delete(0, tk.END)
        edGridZs.delete(0, tk.END)
    # define button command function
    def btDrawPoints_clicked(undistort=False):
        # check ratio button 
        if rbVar.get() == 0: # single-photo calibration
            print("# single-photo calibration")
        else:
            print("# chessboard calibration")
        # get background image
        try:
            fname = edFile.get(0., tk.END)
            # check if it is single-photo calibration or chessboard calibration
            if rbVar.get() == 0: # single-photo calibration
                fname = fname.split()[0]
            else: # chessboard calibration
                imgidx = int(edGridImgIdx.get())
                fname = fname.split()[imgidx-1]
            print("The background file is %s " % fname)
            bkimg = cv.imread(fname)
            if type(bkimg) != type(None) and bkimg.shape[0] > 0:
                print("The image size is %d/%d (w/d)" % (bkimg.shape[1], bkimg.shape[0]))
            else:
                tkCalib_printMessage("# Error: Cannot read background image %s"
                                     % bkimg)
                raise Exception("invalid_image")
        except:
            imgSize = imgSizeFromBt()
            bkimg = np.ones((imgSize[1], imgSize[0], 3), dtype=np.uint8) * 255            
        # get intrinsic parameters
        try:
            # get cmat
            strCmat = edCmat.get(0., tk.END)
            cmat = npFromString(strCmat).reshape(3, 3)
            # get dvec
            strDvec = edDvec.get(0., tk.END)
            dvec = npFromString(strDvec).reshape(1, -1)
        except:
#            tkCalib_printMessage("# Error: Cannot get calibrated result.")
            tk.messagebox.showerror(title="Error", message="# Error: Cannot get calibrated result..")
            return            
        # load image
        imgd = bkimg.copy()
        # get projected points
        try:
            strPrjPoints= edCoordPrj.get(0., tk.END)
            prjPointsAll = npFromString(strPrjPoints).reshape((-1, 2))
            # if it is chessboard calibration, we need to get only a part of prjPointsAll
            if rbVar.get() == 0: # single-photo calibration
                prjPointsThisPhoto = prjPointsAll
            else: # chessboard calibration
                imgidx = int(edGridImgIdx.get())
                nCornersX = int(edCbParams.get(0., tk.END).split()[0])
                nCornersY = int(edCbParams.get(0., tk.END).split()[1])
                prjPointsThisPhoto = prjPointsAll.reshape(-1, nCornersX*nCornersY, 2)[imgidx-1]
            if prjPointsThisPhoto.shape[0] <= 0:
                raise Exception('invalid_proPoints')
            # draw image points
            color=[0,255,255]; 
            markerType=cv.MARKER_SQUARE
            markerSize=max(imgd.shape[0]//100, 3)
            thickness=max(imgd.shape[0]//400, 1)
            lineType=8
            imgd = drawPoints(imgd, prjPointsThisPhoto, color=color, markerType=markerType, 
                              markerSize=markerSize,
                              thickness=thickness, lineType=lineType, savefile='.')
        except:
            tkCalib_printMessage('# No projected points. Skipping plotting them.')
        # draw grid mesh (3D) on the undistorted image
        try:
            strRvec = edRvecs.get(0., tk.END)
            strTvec = edTvecs.get(0., tk.END)
            if rbVar.get() == 0: # single-photo calibration
                rvec = npFromString(strRvec).reshape(3, -1)
                tvec = npFromString(strTvec).reshape(3, -1)
            else:
                rvec = npFromString(strRvec).reshape(-1, 3)[imgidx-1].reshape(3, -1)
                tvec = npFromString(strTvec).reshape(-1, 3)[imgidx-1].reshape(3, -1)
            # get grid Xs Ys Zs
            gridXs = npFromString(edGridXs.get()).reshape((-1)).astype(np.float64)
            gridYs = npFromString(edGridYs.get()).reshape((-1)).astype(np.float64)
            gridZs = npFromString(edGridZs.get()).reshape((-1)).astype(np.float64)
            if gridXs.size > 0 and gridYs.size > 0 and gridZs.size > 0:
                imgd = draw3dMesh(img=imgd, cmat=cmat, dvec=dvec,
                                  rvec=rvec, tvec=tvec,
                                  meshx=gridXs, meshy=gridYs, meshz=gridZs, 
                                  color=(64,255,255), thickness=2, shift=0, 
                                  savefile='.')
        except:
            print("# Skipping plotting grid.")
        # get image points
        try:
            strPointsImg= edCoordImg.get(0., tk.END)
            if rbVar.get() == 0: # single-photo calibration
                points2d = npFromString(strPointsImg).reshape((-1, 2))
            else:                # chessboard calibration
                imgidx = int(edGridImgIdx.get())
                nCornersX = int(edCbParams.get(0., tk.END).split()[0])
                nCornersY = int(edCbParams.get(0., tk.END).split()[1])
                points2d = npFromString(strPointsImg).reshape(-1, nCornersX*nCornersY, 2)[imgidx-1].reshape(-1, 2)
            if points2d.shape[0] <= 0:
                raise Exception('invalid_image_points')
            # draw image points
            color=[0,255,0]; 
            markerType=cv.MARKER_CROSS
            markerSize=max(4, imgd.shape[0] // 100) # 40
            thickness=max(1, imgd.shape[0] // 400) # 10
            lineType=8
            imgd = drawPoints(imgd, points2d, color=color, markerType=markerType, 
                              markerSize=markerSize,
                              thickness=thickness, lineType=lineType, savefile='.')
        except:
            tkCalib_printMessage('# No image point. Skipping plotting image points.')
        # show drawn image on the screen 
#        winW = win.winfo_width()
#        winH = win.winfo_height()
#        imgr = cv.resize(imgd, (winW, winH))
#        cv.imshow("Points", imgr); cv.waitKey(0); 
        imshow2("Points", imgd, winmax=(1600, 800), interp=cv.INTER_LANCZOS4)
        try:
            cv.destroyWindow("Points")
        except:
            None
        # ask if user wants to save the image to file or not
        fname = edFile.get(0., tk.END)
        fname = fname.split()[0]
        initDir = os.path.split(fname)[0]
        ufileDirFile = uiputfile("Save image to file ...", initialDirectory=initDir)
        # if xfile is selected
        if (type(ufileDirFile) == tuple or type(ufileDirFile) == list) and \
            len(ufileDirFile) >= 2 and type(ufileDirFile[0]) == str and \
            len(ufileDirFile[1]) >= 1:
            # save image to selected file
            ufile = os.path.join(ufileDirFile[0], ufileDirFile[1])
            cv.imwrite(ufile, imgd)        
        return
    # set button command function
    btDrawPoints.configure(command=btDrawPoints_clicked)
    
    # define button command function
    def btDrawPointsUndist_clicked():
        btDrawPoints_clicked(undistort=True)


    # set button command function
    btDrawPointsUndist.configure(command=btDrawPointsUndist_clicked)
    # #######################################################
    # Button of [Save camera parameters (rvec/tvec/fx/fy/cx/cy/k1.../tauy)]
    # #######################################################
    btSaveCameraParameters = tk.Button(frame3, text='Save camera parameters', 
                                       width=40, height=1)
    btSaveCameraParameters.pack()
    # define button command function 
    def btSaveCameraParameters_clicked():
        # get parameters
        try:
            # image size
            strImgSize = edImgSize.get(0., tk.END)
            imgSize = npFromString(strImgSize)
            # get rvec
            strRvec = edRvecs.get(0., tk.END)
            rvec = npFromString(strRvec).reshape(3, -1)
            # get tvec
            strTvec = edTvecs.get(0., tk.END)
            tvec = npFromString(strTvec).reshape(3, -1)
            # get cmat
            strCmat = edCmat.get(0., tk.END)
            cmat = npFromString(strCmat).reshape(3, 3)
            # get dvec
            strDvec = edDvec.get(0., tk.END)
            dvec = npFromString(strDvec).reshape(1, -1)
        except:
            tkCalib_printMessage('# Error: btSaveCameraParameters_clicked: Cannot get parameters from edit texts')
        # save to file
        # ask if user wants to save the image to file or not
        fname = edFile.get(0., tk.END)
        # fname could be 1 file or multiple files. We get file [0]
        fname = fname.split()[0]
        initDir = os.path.split(fname)[0]
        ufileDirFile = uiputfile("Save camera parameters to file ...", initialDirectory=initDir)
        # if file is selected
        if (type(ufileDirFile) == tuple or type(ufileDirFile) == list) and \
            len(ufileDirFile) >= 2 and type(ufileDirFile[0]) == str and \
            len(ufileDirFile[1]) >= 1:
            # save image to selected file
            ufile = os.path.join(ufileDirFile[0], ufileDirFile[1])
            writeCamera(ufile, imgSize, rvec, tvec, cmat, dvec)
        #
        return
    # set button command function 
    btSaveCameraParameters.configure(command=btSaveCameraParameters_clicked)
    
    # #######################################################
    # Frame 4
    # #######################################################
    # Button and edit text of [Projected 2D coordinates]
    # #######################################################
    btCoordPrj = tk.Button(frame4, text='Projected coordinates')
    btCoordPrj.pack()
    btCoordPrj.config(state='disabled')
    #   edit text with text height of 5
    edCoordPrj = tk.Text(frame4, width=40, height=3, undo=True, 
                         autoseparators=True, maxundo=-1)
    edCoordPrj.pack()
    #   set initial text for demonstration
    edCoordPrj.delete(0., tk.END)
    # #######################################################
    # Button and edit text of [Projected errors]
    # #######################################################
    btPrjErrors = tk.Button(frame4, text='Projected errors')
    btPrjErrors.pack()
    #   edit text with text height of 5
    edPrjErrors = tk.Text(frame4, width=40, height=3, undo=True, 
                          autoseparators=True, maxundo=-1)
    edPrjErrors.pack()
    #   set initial text for demonstration
    edPrjErrors.delete(0., tk.END)    
    # #######################################################
    # Button of [Rvecs (calculated)]
    # #######################################################
    btRvecs = tk.Button(frame4, text='Rvecs.')
    btRvecs.pack()
    edRvecs = tk.Text(frame4, width=40, height=2, undo=True, 
                      autoseparators=True, maxundo=-1)
    edRvecs.pack()
    # #######################################################
    # Button of [Tvecs (calculated)]
    # #######################################################
    # Calculated rvecs
    btTvecs = tk.Button(frame4, text='Tvecs.')
    btTvecs.pack()
    edTvecs = tk.Text(frame4, width=40, height=2, undo=True, 
                      autoseparators=True, maxundo=-1)
    edTvecs.pack()
    # #######################################################
    # Button of [Camera mat (calculated)]
    # #######################################################
    # Calculated camera matrix 
    btCmat = tk.Button(frame4, text='Camera mat')
    btCmat.pack()
    edCmat = tk.Text(frame4, width=40, height=2, undo=True, 
                     autoseparators=True, maxundo=-1)
    edCmat.pack()
    # #######################################################
    # Button of [Distortion coeff. (calculated)]
    # #######################################################
    # Calculated distortion vector 
    btDvec = tk.Button(frame4, text='Distortion coeff.')
    btDvec.pack()
    edDvec = tk.Text(frame4, width=40, height=2, undo=True, 
                     autoseparators=True, maxundo=-1)
    edDvec.pack()
    # #######################################################
    # Button of [Camera positions (calculated)]
    # #######################################################
    btCampos = tk.Button(frame4, text='Camera position(s)')
    btCampos.pack()
    edCampos = tk.Text(frame4, width=40, height=2, undo=True, 
                       autoseparators=True, maxundo=-1)
    edCampos.pack()
    # #######################################################
    # Button of [Parameters all-in-one-column]
    # #######################################################
    btOneCol = tk.Button(frame4, text='All-in-one-column')
    btOneCol.pack()
    edOneCol = tk.Text(frame4, width=40, height=2, undo=True, 
                       autoseparators=True, maxundo=-1)
    edOneCol.pack()
    # #######################################################
    # function of close()
    # #######################################################
    def winClose():
        # nonlocal
        nonlocal imgSize, rvec, tvec, cmat, dvec
        # save current edCoord3d
        try:
            strPoints3d = edCoord3d.get(0., tk.END)
            matPoints3d = npFromString(strPoints3d).reshape((-1,3))
            matPoints3d = matPoints3d.astype(np.float32)
            np.savetxt(os.path.join(os.getcwd(), 'tkCalib_init_coord3d.npy'), matPoints3d)
        except:
            tkCalib_printMessage("# Error: Cannot parse 3D coord.")
        # save current edCoordImg
        try:
            strPointsImg= edCoordImg.get(0., tk.END)
            matPoints2d = npFromString(strPointsImg).reshape((-1,2))
            matPoints2d = matPoints2d.astype(np.float32)
            np.savetxt(os.path.join(os.getcwd(), 'tkCalib_init_coordImg.npy'), matPoints2d)
        except:
            tkCalib_printMessage("# Error: Cannot parse image (2D) coord.")
        # save current calibration image file
        try:
            strCalibImg = edFile.get(0., tk.END)
            tfile = open(os.path.join(os.getcwd(), 'tkCalib_init_imgFile.txt'), "w")
            n = tfile.write(strCalibImg)
            tfile.close()
        except:
            tkCalib_printMessage("# Error: Cannot get calibration image file path")
        # save current image size
        try:
            theStr = edImgSize.get(0., tk.END)
            imgSize = npFromString(theStr).reshape((1,-1))
            imgSize = imgSize.astype(int)
            np.savetxt(os.path.join(os.getcwd(), 'tkCalib_init_imgSize.npy'), imgSize, fmt='%d')
        except:
            tkCalib_printMessage("# Error: Cannot parse image size")
        # save current cmatGuess
        try:
            theStr = edCmatGuess.get(0., tk.END)
            theMat = npFromString(theStr).reshape((3,3))
            theMat = theMat.astype(np.float64)
            np.savetxt(os.path.join(os.getcwd(), 'tkCalib_init_cmatGuess.npy'), theMat)
        except:
            tkCalib_printMessage("# Error: Cannot parse camera mat (guessed).")
        # save current dvecGuess
        try:
            theStr = edDvecGuess.get(0., tk.END)
            theMat = npFromString(theStr).reshape((-1))
            theMat = theMat.astype(np.float64)
            np.savetxt(os.path.join(os.getcwd(), 'tkCalib_init_dvecGuess.npy'), theMat)
        except:
            tkCalib_printMessage("# Error: Cannot parse distortion vec (guessed).")
        # save current chessboard parameters
        try:
            theStr = edCbParams.get(0., tk.END)
            theMat = npFromString(theStr).reshape((-1))
            theMat = theMat.astype(np.float64)
            np.savetxt(os.path.join(os.getcwd(), 'tkCalib_init_cboardParams.npy'), theMat)
        except:
            tkCalib_printMessage("# Error: Cannot parse calibration board parameters (nCornersX nCornersY dxCorners dyCorners.")
        # save current flags
        try:
            theStr = edFlags.get()
            theStr = theStr[theStr.find(':')+1:]
            theMat = npFromString(theStr).reshape((-1))
            theMat = theMat.astype(np.int32)
            np.savetxt(os.path.join(os.getcwd(), 'tkCalib_init_calibFlags.npy'), theMat, fmt='%i')
        except:
            tkCalib_printMessage("# Error: Cannot parse calibration flags.")
        # save current gridXs/Ys/Zs
        try:
            theStr = edGridXs.get()
            theMat = npFromString(theStr).reshape((-1)).astype(np.float64)
            np.savetxt(os.path.join(os.getcwd(), 'tkCalib_init_gridXs.npy'), theMat)
            theStr = edGridYs.get()
            theMat = npFromString(theStr).reshape((-1)).astype(np.float64)
            np.savetxt(os.path.join(os.getcwd(), 'tkCalib_init_gridYs.npy'), theMat)
            theStr = edGridZs.get()
            theMat = npFromString(theStr).reshape((-1)).astype(np.float64)
            np.savetxt(os.path.join(os.getcwd(), 'tkCalib_init_gridZs.npy'), theMat)
        except:
            tkCalib_printMessage("# Error: Cannot parse grid coordinates (Xs/Ys/Zs).")
                
        print("tkCalib window closed.")
        imgSize = npFromString(edImgSize.get(0., tk.END)).astype(int).flatten()
        try:
            rvec = npFromString(edRvecs.get(0., tk.END)).reshape(3, -1)
        except:
            rvec = np.array([], dtype=float)
        try:
            tvec = npFromString(edTvecs.get(0., tk.END)).reshape(3, -1)
        except:
            tvec = np.array([], dtype=float)
        try:
            cmat = npFromString(edCmat.get(0., tk.END)).reshape((3, 3))
        except:
            cmat = np.array([], dtype=float)
        try:
            dvec = npFromString(edDvec.get(0., tk.END)).reshape((1,-1))
        except:
            dvec = np.array([], dtype=float)
        win.destroy()
        return     
    
    rb_clicked()
    win.lift()
    win.attributes("-topmost", True)
    win.after_idle(win.attributes,'-topmost',False)
    win.protocol("WM_DELETE_WINDOW", winClose)

    win.mainloop()
#    win.destroy()


if __name__ == "__main__":
#    camParams = tkCalib()
    tkCalib()