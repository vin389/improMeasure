import os, sys
import time, datetime
import tkinter.filedialog
import tkinter.scrolledtext
import tkinter as tk
import numpy as np
import cv2 as cv

from improCalib import newArrayByMapping, countCalibPoints
from improMisc import uigetfile, uiputfile
from improStrings import npFromString, stringFromNp, npFromTupleNp
from drawPoints import drawPoints
from imshow2 import imshow2
from writeCamera import writeCamera
from draw3dMesh import draw3dMesh

def tkCalib_printMessage(msg: str):
    strfNow = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("%s: %s" % (strfNow, msg))
    return


def tkCalib():
    camParams = np.zeros(14, dtype=np.float64)
    
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

    # #######################################################
    # Button and edit text of [3D coordinates (World coord.)]
    # #######################################################
    # 3D coordinates
    #   button '3D coordinates' as a label
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
    
    btCoord3d = tk.Button(win, text='3D coordinates (World coord.)', \
                          command=btCoord3d_clicked)
    btCoord3d.grid(row=0, column=0)
    #   edit text with text height of 5
    edCoord3d = tk.scrolledtext.ScrolledText(win, width=40, height=4, undo=True, autoseparators=True, maxundo=-1)
    edCoord3d.grid(row=1, rowspan=4, column=0)
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

    # #######################################################
    # Button and edit text of [Image 2D coordinates]
    # #######################################################
    # Image (2D) coordinates
    #   button 'Image 2D coordinates' as a label
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
    btCoordImg = tk.Button(win, text='Image 2D coordinates',\
                           command=btCoordImg_clicked)
    btCoordImg.grid(row=5, column=0)
    #   edit text with text height of 5
    edCoordImg = tk.scrolledtext.ScrolledText(win, width=40, height=3, undo=True, autoseparators=True, maxundo=-1)
    edCoordImg.grid(row=6, rowspan=3, column=0)
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

    
    # #######################################################
    # Button of [Select calibration image ...]
    # #######################################################
    # Calibration image
    #   file path text
    edFile = tk.Entry(win, width=40)
    edFile.grid(row=10, column=0)
    #   set initial text for demonstration
    try:
        tfile = open(os.path.join(os.getcwd(), 'tkCalib_init_imgFile.txt'), "r")
        fname = tfile.read()
        tfile.close()
        edFile.delete(0, tk.END)
        edFile.insert(0, fname)
    except:
        edFile.insert(0, 'c:/images/calibration.bmp')
        
    #   btFile 'Select image ...'
    def btFile_clicked():
        # select file
        initDir = os.path.split(edFile.get())[0]
        ufileDirFile = uigetfile(initialDirectory=initDir)
        # if xfile is selected
        if (type(ufileDirFile) == tuple or type(ufileDirFile) == list) and \
            len(ufileDirFile) >= 2 and type(ufileDirFile[0]) == str:
            # display full path of the selected file
            edFile.delete(0, tk.END)
            ufile = os.path.join(ufileDirFile[0], ufileDirFile[1])
            edFile.insert(0, ufile)
            btImgSize_clicked()
            # # try to get the image size and display to edImgSize
            # try:
            #     img = cv.imread(ufile)
            #     if type(img) != type(None) and img.shape[0] >= 1:
            #         strSize = "%d  %d" % (img.shape[1], img.shape[0])
            #         edImgSize.delete(0, tk.END)
            #         edImgSize.insert(0, strSize)
            #     else:
            #         edImgSize.delete(0, tk.END)
            #         edImgSize.insert(0, "(cannot open the file as an image)")
            # except:
            #     edImgSize.delete(0, tk.END)
            #     edImgSize.insert(0, "(had exception when opening the file as an image)")
                
        return
    btFile = tk.Button(win, text='Select calibration image ...', 
                       command=btFile_clicked)
    btFile.grid(row=9, column=0)

    # #######################################################
    # Button of [Image size:]
    # #######################################################
    # Image size imgW imgH 
    #   text (entry) of the image size imgW imgH
    edImgSize = tk.Entry(win, width=40)
    edImgSize.grid(row=12, column=0)
    #   set initial text for demonstration
    try:
        theMat = np.loadtxt(os.path.join(os.getcwd(), 'tkCalib_init_imgSize.npy')).astype(int)
        theStr = stringFromNp(theMat, sep='\t')
        edImgSize.delete(0, tk.END)
        edImgSize.insert(0, theStr)
    except:
        tkCalib_printMessage("# Warning: Cannot parse tkCalib_init_imgSize.npy")
        edImgSize.delete(0, tk.END)
        edImgSize.insert(0, '1920 1080')

    def btImgSize_clicked():
        # try to open the image file and check the image size
        try:
            fname = edFile.get()
            img = cv.imread(fname)
            if type(img) == np.ndarray and img.shape[0] >= 1:
                imgH = img.shape[0]
                imgW = img.shape[1]
                edImgSize.delete(0, tk.END)
                edImgSize.insert(0, '%d %d' % (imgW, imgH))
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
            print("Cannot read the image file.")
        strImgSize = edImgSize.get()
        print('Image size: ', strImgSize)
        return
    def imgSizeFromBt():
        strImgSize = edImgSize.get()
        imgSize = npFromString(strImgSize)
        if type(imgSize) != np.ndarray or imgSize.size != 2:
            print("# error: Image size is invalid (edit text is %s)" % (strImgSize))
            return None
        return imgSize

    btImgSize = tk.Button(win, text='Image size: width height',
                          command=btImgSize_clicked)
    btImgSize.grid(row=11, column=0)
#    btImgSize['state'] = tk.DISABLED

    # #######################################################
    # Button of [Camera mat (init guess)]
    # #######################################################
    # Initial guess of camera matrix 
    edCmatGuess = tk.scrolledtext.ScrolledText(win, width=40, height=2, undo=True, autoseparators=True, maxundo=-1)
    edCmatGuess.grid(row=14, rowspan=2, column=0)
    #   set initial text for demonstration
    try:
        theMat = np.loadtxt(os.path.join(os.getcwd(), 'tkCalib_init_cmatGuess.npy'))
        theStr = stringFromNp(theMat, sep='\t')
        edCmatGuess.delete(0., tk.END)
        edCmatGuess.insert(0., theStr)
    except:
        print("Warning: Cannot load tkCalib_init_cmatGuess.npy")
        edCmatGuess.delete(0., tk.END)
        edCmatGuess.insert(0., ' 5000. 0 0 \n 0 5000. 0 \n 1999.5 1999.5 1')
    #   button 'Camera mat (init guess)' as a label
    def btCmatGuess_clicked():
        try:
            strCmatGuess = edCmatGuess.get(0., tk.END)
            cmatGuess = npFromString(strCmatGuess).reshape((3,3))
        except:
            cmatGuess = np.array([])
        print('Initial guess of camera matrix:\n', cmatGuess)
        return cmatGuess
    btCmatGuess = tk.Button(win, text='Camera mat (init guess)',\
                           command=btCmatGuess_clicked)
    btCmatGuess.grid(row=13, column=0)

    # #######################################################
    # Button of [Distortion coeff. (init guess)]
    # #######################################################
    # Initial guess of distortion vector 
    #   text (entry) of the image size imgW imgH
    edDvecGuess = tk.scrolledtext.ScrolledText(win, width=40, height=2, undo=True, autoseparators=True, maxundo=-1)
    edDvecGuess.grid(row=17, rowspan=2, column=0)
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

    #   button 'Camera mat (init guess)' as a label
    def btDvecGuess_clicked():
        strDvecGuess = edDvecGuess.get(0., tk.END)
        try:
            cdvecGuess = npFromString(strDvecGuess).reshape((1,-1))
        except:
            cdvecGuess = np.array([])
        print('Initial guess of distortion vector:\n', cdvecGuess)
        return cdvecGuess
    btDvecGuess = tk.Button(win, text='Distortion coeff. (init guess)',\
                           command=btDvecGuess_clicked)
    btDvecGuess.grid(row=16, column=0)

    # #######################################################
    # Checkbuttons of calibration flags
    # #######################################################
    # Checkbuttons of calibration flags 
    #   allowing user to switch on/off of every flag
    #   once clicked, sum of flags is displaced 
    # init of flags 
    edFlags = tk.Entry(win)
    edFlags.grid(row=0, column=1)
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
    # create checkbutton for flags
    for i in range(len(strFlags)):
        # generate statement string for creating checkbutton
        #   E.g., for i == 0
        #   "tk.Checkbutton(win, text='CALIB_USE_INTRINSIC_GUESS (0)',
        #                   command=ck_clicked, variable=ckValues[i])"
        evalStr = "tk.Checkbutton(win, text='"
        evalStr += strFlags[i] + " (%d)" % (eval('cv.' + strFlags[i])) + "', "
        evalStr += "command=ck_clicked, "
        evalStr += "variable=ckValues[%d]" % (i) + ")"
        # create a tk.IntVar() for checkbutton value
        ckValues.append(tk.IntVar())
        # create a checkbutton
        ckFlags.append(eval(evalStr)) # 
        # position a checkbutton
        ckFlags[i].grid(row=i+1, column=1, sticky='W')
    # set default flags: 
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
    # GUI grid column 2
    # #######################################################
    c2row = 0
    # #######################################################
    # Button of [Calibrate camera]
    # #######################################################
    # calibrate camera button
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
        #　ｇｅｔ　ｔｈｅ　ｍａｐｐｉｎｇ　
        validsOfPoints3d, validsOfPoints2d, validCalibPoints, \
            idxAllToValid, idxValidToAll, validPoints3d, validPoints2d = \
            countCalibPoints(points3d, points2d)
        # variables and parameters to run camera calibration 
        nCalPoints = validPoints3d.shape[0]
        objPoints = validPoints3d.reshape((1, nCalPoints, 3)).astype(np.float32)
        imgPoints = validPoints2d.reshape((1, nCalPoints, 2)).astype(np.float32)
        imgSize = imgSizeFromBt().astype(int)
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
        # calculate projection errors
        imgPointsValid = imgPoints.reshape((-1, 2))
        prjErrorsValid = (prjPointsValid - imgPointsValid)
        # display
        prjPointsAll = newArrayByMapping(prjPointsValid, idxAllToValid)
        edCoordPrj.delete(0., tk.END)
        edCoordPrj.insert(0., stringFromNp(prjPointsAll, sep='\t'))
        prjErrorsAll = newArrayByMapping(prjErrorsValid, idxAllToValid)
        edPrjErrors.delete(0., tk.END)
        edPrjErrors.insert(0., stringFromNp(prjErrorsAll, sep='\t'))
        # all-in-one-column:rvec/tvec/cmat9/dvec14/empty/campos
        vec29 = np.zeros((29, 1), dtype=float)
        vec29[0:3, 0] = rvecs[0].reshape((-1, 1))[0:3, 0]
        vec29[3:6, 0] = tvecs[0].reshape((-1, 1))[0:3, 0]
        vec29[6:15, 0] = cmat.reshape((9, 1))[0:9, 0]
        lenDvec = dvec.size
        vec29[15:15+lenDvec, 0] = dvec.reshape((lenDvec, 1))[0:lenDvec, 0]
        strOneCol = stringFromNp(vec29, sep='\t') + "\n\n"
        strOneCol += stringFromNp(campos.reshape((-1, 1)), sep='\t')
        edOneCol.delete(0., tk.END)
        edOneCol.insert(0., strOneCol)
        return
    
    btCalib = tk.Button(win, text='Calibrate camera', width=40, height=2,
                          command=btCalib_clicked)
    btCalib.grid(row=c2row, column=2, rowspan=2); c2row += 2

    # #######################################################
    # Button of [Draw projection image]
    # #######################################################
    # 
    def btDrawPoints_clicked():
        # get background image
        fname = edFile.get()
        print("The background file is %s " % fname)
        bkimg = cv.imread(fname)
        if type(bkimg) != type(None) and bkimg.shape[0] > 0:
            print("The image size is %d/%d (w/d)" % (bkimg.shape[1], bkimg.shape[0]))
        else:
            tkCalib_printMessage("# Error: Cannot read background image %s"
                                 % bkimg)
        # get image points
        strPointsImg= edCoordImg.get(0., tk.END)
        try:
            points2d = npFromString(strPointsImg).reshape((-1, 2))
            # draw image points
            imgd = bkimg.copy()
            color=[0,255,0]; 
            markerType=cv.MARKER_CROSS
            markerSize=40
            thickness=4
            lineType=8
            imgd = drawPoints(imgd, points2d, color=color, markerType=markerType, 
                              markerSize=markerSize,
                              thickness=thickness, lineType=lineType, savefile='.')
        except:
            tkCalib_printMessage('# Error: Cannot parse 2D points (size: %d)')
        # get proected points
        strPrjPoints= edCoordPrj.get(0., tk.END)
        try:
            prjPointsAll = npFromString(strPrjPoints).reshape((-1, 2))
            if prjPointsAll.shape[0] <= 0:
                raise Exception
            # draw image points
            color=[0,255,255]; 
            markerType=cv.MARKER_SQUARE
            markerSize=60
            thickness=4
            lineType=8
            imgd = drawPoints(imgd, prjPointsAll, color=color, markerType=markerType, 
                              markerSize=markerSize,
                              thickness=thickness, lineType=lineType, savefile='.')
        except:
            tkCalib_printMessage('# Error: Cannot parse 2D projected points')
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
        initDir = os.path.split(edFile.get())[0]
        ufileDirFile = uiputfile("Save image to file ...", initialDirectory=initDir)
        # if xfile is selected
        if (type(ufileDirFile) == tuple or type(ufileDirFile) == list) and \
            len(ufileDirFile) >= 2 and type(ufileDirFile[0]) == str and \
            len(ufileDirFile[1]) >= 1:
            # save image to selected file
            ufile = os.path.join(ufileDirFile[0], ufileDirFile[1])
            cv.imwrite(ufile, imgd)        
        return
    btDrawPoints = tk.Button(win, text='Draw points', width=40, height=1,
                          command=btDrawPoints_clicked)
    btDrawPoints.grid(row=c2row, column=2, rowspan=1); c2row += 1

    # #######################################################
    # Text of grid coordinates (Xs, Ys, and Zs)
    # #######################################################
    edGridXs = tk.Entry(win, width=40)
    edGridXs.grid(row=c2row, column=2, rowspan=1); c2row += 1;
    edGridYs = tk.Entry(win, width=40)
    edGridYs.grid(row=c2row, column=2, rowspan=1); c2row += 1;
    edGridZs = tk.Entry(win, width=40)
    edGridZs.grid(row=c2row, column=2, rowspan=1); c2row += 1;
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


    # #######################################################
    # Button of [Draw projection image on an undistorted image]
    # #######################################################
    # 
    def btDrawPointsUndistort_clicked():
        # get background image
        fname = edFile.get()
        print("The background file is %s " % fname)
        bkimg = cv.imread(fname)
        if type(bkimg) != type(None) and bkimg.shape[0] > 0:
            print("The image size is %d/%d (w/d)" % (bkimg.shape[1], bkimg.shape[0]))
        else:
            tkCalib_printMessage("# Error: Cannot read background image %s"
                                 % bkimg)
        # get image points
        try:
            # get cmat
            strCmat = edCmat.get(0., tk.END)
            cmat = npFromString(strCmat).reshape(3, 3)
            # get dvec
            strDvec = edDvec.get(0., tk.END)
            dvec = npFromString(strDvec).reshape(1, -1)
            # undistort image
            imgud = cv.undistort(bkimg, cmat, dvec)             
        except:
            tkCalib_printMessage('# Error: Cannot undistort image')
            return
        try:
            # get image points
            strPointsImg= edCoordImg.get(0., tk.END)
            points2d = npFromString(strPointsImg).reshape((-1, 2))
            # get undistorted image points
            points2du = cv.undistortPoints(points2d, cmat, dvec).reshape(-1, 2)
            nPoints = points2d.shape[0]
            points2duh = np.ones((nPoints, 3), dtype=float)
            points2duh[:, 0:2] = points2du
            points2duh = np.matmul(cmat, points2duh.transpose()).transpose()
            points2du = points2duh[:, 0:2]
        except:
            tkCalib_printMessage('# Error: Cannot get image points')
            return
        try:
            # draw image points
            color=[0,255,0]; 
            markerType=cv.MARKER_CROSS
            markerSize=40
            thickness=4
            lineType=8
            imgud = drawPoints(imgud, points2du, color=color, markerType=markerType, 
                              markerSize=markerSize,
                              thickness=thickness, lineType=lineType, savefile='.')
        except:
            tkCalib_printMessage('# Error: Cannot draw undistorted image points on undistorted image')
        # get proected points
        try:
            strPrjPoints= edCoordPrj.get(0., tk.END)
            prjPointsAll = npFromString(strPrjPoints).reshape((-1, 2))
            if prjPointsAll.shape[0] <= 0:
                raise Exception
            # get undistorted image points
            prjPointsuAll = cv.undistortPoints(prjPointsAll, cmat, dvec).reshape(-1, 2)
            nPoints = prjPointsuAll.shape[0]
            prjPointsuhAll = np.ones((nPoints, 3), dtype=float)
            prjPointsuhAll[:, 0:2] = prjPointsuAll
            prjPointsuhAll = np.matmul(cmat, prjPointsuhAll.transpose()).transpose()
            prjPointsuAll = prjPointsuhAll[:, 0:2]
            # draw image points
            color=[0,255,255]; 
            markerType=cv.MARKER_SQUARE
            markerSize=60
            thickness=4
            lineType=8
            imgd = drawPoints(imgud, prjPointsuAll, color=color, markerType=markerType, 
                              markerSize=markerSize,
                              thickness=thickness, lineType=lineType, savefile='.')
        except:
            tkCalib_printMessage('# Error: Cannot draw undistorted projected points')
        # draw grid mesh (3D) on the undistorted image
        strRvec = edRvecs.get(0., tk.END)
        rvec = npFromString(strRvec).reshape(3, -1)
        # get tvec
        strTvec = edTvecs.get(0., tk.END)
        tvec = npFromString(strTvec).reshape(3, -1)
        # get grid Xs Ys Zs
        gridXs = npFromString(edGridXs.get()).reshape((-1)).astype(np.float64)
        gridYs = npFromString(edGridYs.get()).reshape((-1)).astype(np.float64)
        gridZs = npFromString(edGridZs.get()).reshape((-1)).astype(np.float64)
        imgd = draw3dMesh(img=imgd, cmat=cmat, dvec=dvec, rvec=rvec, tvec=tvec,
               meshx=gridXs, meshy=gridYs, meshz=gridZs, 
               color=(64,255,64), thickness=4, shift=0, 
               savefile='.')
        
        # show drawn image on the screen 
#        winW = win.winfo_width()
#        winH = win.winfo_height()
#        imgr = cv.resize(imgud, (winW, winH))
#        cv.imshow("Points on undistorted Image", imgr); cv.waitKey(0); 
        imshow2("Points of undistorted Image", imgd, winmax=(1600, 800), interp=cv.INTER_LANCZOS4)
        try:
            cv.destroyWindow("Points on undistorted Image")
        except:
            None
        # ask if user wants to save the image to file or not
        initDir = os.path.split(edFile.get())[0]
        ufileDirFile = uiputfile("Save undistorted image to file ...", initialDirectory=initDir)
        # if xfile is selected
        if (type(ufileDirFile) == tuple or type(ufileDirFile) == list) and \
            len(ufileDirFile) >= 2 and type(ufileDirFile[0]) == str and \
            len(ufileDirFile[1]) >= 1:
            # save image to selected file
            ufile = os.path.join(ufileDirFile[0], ufileDirFile[1])
            cv.imwrite(ufile, imgud)        
        return
    btDrawPointsUndist = tk.Button(win, text='Draw points (undistorted)', width=40, height=1,
                          command=btDrawPointsUndistort_clicked)
    btDrawPointsUndist.grid(row=c2row, column=2, rowspan=1); c2row += 1

    # #######################################################
    # Button of [Save camera parameters (rvec/tvec/fx/fy/cx/cy/k1.../tauy)]
    # #######################################################
    def btSaveCameraParameters_clicked():
        # get parameters
        try:
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
        initDir = os.path.split(edFile.get())[0]
        ufileDirFile = uiputfile("Save camera parameters to file ...", initialDirectory=initDir)
        # if file is selected
        if (type(ufileDirFile) == tuple or type(ufileDirFile) == list) and \
            len(ufileDirFile) >= 2 and type(ufileDirFile[0]) == str and \
            len(ufileDirFile[1]) >= 1:
            # save image to selected file
            ufile = os.path.join(ufileDirFile[0], ufileDirFile[1])
            writeCamera(ufile, rvec, tvec, cmat, dvec)
        #
        return
    btSaveCameraParameters = tk.Button(win, text='Save camera parameters', width=40, height=1,
                          command=btSaveCameraParameters_clicked)
    btSaveCameraParameters.grid(row=c2row, column=2, rowspan=1); c2row += 1

    # #######################################################
    # Button and edit text of [Projected 2D coordinates]
    # #######################################################
    # Projected image (2D) coordinates
    #   button 'Projected 2D coordinates' as a label
    def btCoordPrj_clicked():
        return
    btCoordPrj = tk.Button(win, text='Projected coordinates',\
                           command=btCoordPrj_clicked)
    btCoordPrj.grid(row=0, column=3)
    #   edit text with text height of 5
    edCoordPrj = tk.scrolledtext.ScrolledText(win, width=40, height=3, undo=True, autoseparators=True, maxundo=-1)
    edCoordPrj.grid(row=1, rowspan=3, column=3)
    #   set initial text for demonstration
    edCoordPrj.delete(0., tk.END)
    
    # #######################################################
    # Button and edit text of [Projected errors]
    # #######################################################
    # Projected errors
    #   button 'Projected errors' as a label
    def btPrjErrors_clicked():
        return
    btPrjErrors = tk.Button(win, text='Projected errors',\
                           command=btPrjErrors_clicked)
    btPrjErrors.grid(row=4, column=3)
    #   edit text with text height of 5
    edPrjErrors = tk.scrolledtext.ScrolledText(win, width=40, height=3, undo=True, autoseparators=True, maxundo=-1)
    edPrjErrors.grid(row=5, rowspan=3, column=3)
    #   set initial text for demonstration
    edPrjErrors.delete(0., tk.END)

    # #######################################################
    # Button of [Rvecs (calculated)]
    # #######################################################
    # Calculated rvecs
    edRvecs = tk.scrolledtext.ScrolledText(win, width=40, height=2, undo=True, autoseparators=True, maxundo=-1)
    edRvecs.grid(row=9, rowspan=2, column=3)
    #   button 'Rvecs' as a label
    def btRvecs_clicked():
        return 
    btRvecs = tk.Button(win, text='Rvecs.',\
                           command=btRvecs_clicked)
    btRvecs.grid(row=8, column=3)

    # #######################################################
    # Button of [Tvecs (calculated)]
    # #######################################################
    # Calculated rvecs
    edTvecs = tk.scrolledtext.ScrolledText(win, width=40, height=2, undo=True, autoseparators=True, maxundo=-1)
    edTvecs.grid(row=12, rowspan=2, column=3)
    #   button 'Tvecs' as a label
    def btTvecs_clicked():
        return 
    btTvecs = tk.Button(win, text='Tvecs.',\
                           command=btTvecs_clicked)
    btTvecs.grid(row=11, column=3)


    # #######################################################
    # Button of [Camera mat (calculated)]
    # #######################################################
    # Calculated camera matrix 
    edCmat = tk.scrolledtext.ScrolledText(win, width=40, height=2, undo=True, autoseparators=True, maxundo=-1)
    edCmat.grid(row=15, rowspan=2, column=3)
    #   button 'Camera mat (init guess)' as a label
    def btCmat_clicked():
        return 
    btCmat = tk.Button(win, text='Camera mat',\
                           command=btCmat_clicked)
    btCmat.grid(row=14, column=3)

    # #######################################################
    # Button of [Distortion coeff. (calculated)]
    # #######################################################
    # Calculated distortion vector 
    edDvec = tk.scrolledtext.ScrolledText(win, width=40, height=2, undo=True, autoseparators=True, maxundo=-1)
    edDvec.grid(row=18, rowspan=2, column=3)
    #   button 'Camera mat (init guess)' as a label
    def btDvec_clicked():
        return 
    btDvec = tk.Button(win, text='Distortion coeff.',\
                           command=btDvec_clicked)
    btDvec.grid(row=17, column=3)

    # #######################################################
    # Button of [Camera positions (calculated)]
    # #######################################################
    # Calculated camera positions
    edCampos = tk.scrolledtext.ScrolledText(win, width=40, height=2, undo=True, autoseparators=True, maxundo=-1)
    edCampos.grid(row=21, rowspan=2, column=3)
    #   button 'Tvecs' as a label
    def btCampos_clicked():
        return 
    btCampos = tk.Button(win, text='Camera position(s)',\
                           command=btCampos_clicked)
    btCampos.grid(row=20, column=3)
    
    # #######################################################
    # Button of [All-in-one-column]
    # #######################################################
    # Calculated camera positions
    edOneCol = tk.scrolledtext.ScrolledText(win, width=40, height=2, undo=True, autoseparators=True, maxundo=-1)
    edOneCol.grid(row=24, rowspan=2, column=3)
    #   button 'Tvecs' as a label
    def btOneCol_clicked():
        return 
    btOneCol = tk.Button(win, text='All-in-one-column',\
                           command=btOneCol_clicked)
    btOneCol.grid(row=23, column=3)

    # #######################################################
    # Button of [Close]
    # #######################################################
    def winClose():
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
            strCalibImg = edFile.get()
            tfile = open(os.path.join(os.getcwd(), 'tkCalib_init_imgFile.txt'), "w")
            n = tfile.write(strCalibImg)
            tfile.close()
        except:
            tkCalib_printMessage("# Error: Cannot get calibration image file path")
        # save current image size
        try:
            theStr = edImgSize.get()
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
                
        print("Window Closed")
        
        win.destroy()
        return
    
    # main window         
    win.lift()
    win.attributes("-topmost", True)
    win.after_idle(win.attributes,'-topmost',False)
    win.protocol("WM_DELETE_WINDOW", winClose)
    win.mainloop()
#    win.destroy()

    return

    
    
    
    
    
    
    
    
    
    
    
camParams = tkCalib()
