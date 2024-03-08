# This Python file uses the following encoding: utf-8
import os, sys, time

from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog
from PySide6 import QtGui, QtCore
import glob

from createFileList import createFileList
from ui_form import Ui_improMeasure
from improMisc import *

import cv2 as cv
import numpy as np
import pandas as pd
import openpyxl

def cmbCbCalibCam_currentIndexChanged(self):
    # This code does not really run. It is only for intellisense 
    # so that editor recognizes the functions and variables of the self. 
    if type(self) == type(None):
        self.ui = Ui_improMeasure()
    self.ui.pteMsg.appendHtml("<p style=\"color:blue;\">Called cmbCbCalibCam_currentIndexChanged()")
    # if index >=1, change source directory and display calibration photo list
    iCam = self.ui.cmbCbCalibCam.currentIndex()
    if iCam <= 0:
        self.ui.edCbCalibSrcDir.clear()
        self.ui.pteCbCalibSrcPhotos.clear()
    else:
        # get rootDir, cbCalibDir, and cbCalibFiles
        rootDir = self.ui.edRootdir.text()
        cbCalibDir = os.path.join(rootDir, 'calibration_cboard_%d' % (iCam))
        cbCalibFiles_all = glob.glob(os.path.join(cbCalibDir, '*'))
        # display cbCalibDir 
        self.ui.edCbCalibSrcDir.setText(cbCalibDir)
          # if no file, return
        if type(cbCalibFiles_all) != list or len(cbCalibFiles_all) <= 0 \
            or type(cbCalibFiles_all[0]) != str:
            self.ui.pteMsg.appendHtml("<p style=\"color:red;\">Cannot find calibration file in %s" % cbCalibDir)
            return
        cbCalibFiles = imreadSupportedFiles(cbCalibFiles_all)
        if type(cbCalibFiles) != list or len(cbCalibFiles) <= 0 \
            or type(cbCalibFiles[0]) != str:
            self.ui.pteMsg.appendHtml("<p style=\"color:red;\">Cannot find calibration file in %s" % cbCalibDir)
            return
        # display source files of cbCalib
        nfiles = len(cbCalibFiles)
        self.ui.pteCbCalibSrcPhotos.clear()
        for i in range(nfiles):
            self.ui.pteCbCalibSrcPhotos.appendHtml("<p style=\"color:blue;\">" +
                cbCalibFiles[i])
        return        


def edCbCalibSrcDir_returnPressed(self):
    # This code does not really run. It is only for intellisense 
    # so that editor recognizes the functions and variables of the self. 
    if type(self) == type(None):
        self.ui = Ui_improMeasure()
    self.ui.pteMsg.appendHtml("<p style=\"color:blue;\">Called edCbCalibSrcDir_returnPressed()")
    # if camera index >= 1, call cmbCbCalibCam_currentIndexChanged() and return.
    iCam = self.ui.cmbCbCalibCam.currentIndex()
    if iCam >= 1:
        cmbCbCalibCam_currentIndexChanged(self)
        return
    # get the user defined directory of source photos
    cbCalibDir = self.ui.edCbCalibSrcDir.text()
    # if directory does not exist, clear file list and return
    if os.path.exists(cbCalibDir) == False:
        self.ui.pteCbCalibSrcPhotos.clear()
        return
    # list all files
    cbCalibFiles_all = glob.glob(os.path.join(cbCalibDir,'*'))
    # if no file, return
    if type(cbCalibFiles_all) != list or len(cbCalibFiles_all) <= 0 \
        or type(cbCalibFiles_all[0]) != str:
        self.ui.pteMsg.appendHtml("<p style=\"color:red;\">Cannot find calibration file in %s" % cbCalibDir)
        return
    cbCalibFiles = imreadSupportedFiles(cbCalibFiles_all)
    if type(cbCalibFiles) != list or len(cbCalibFiles) <= 0 \
        or type(cbCalibFiles[0]) != str:
        self.ui.pteMsg.appendHtml("<p style=\"color:red;\">Cannot find calibration file in %s" % cbCalibDir)
        return
    # display source files of cbCalib
    nfiles = len(cbCalibFiles)
    self.ui.pteCbCalibSrcPhotos.clear()
    for i in range(nfiles):
        self.ui.pteCbCalibSrcPhotos.appendHtml("<p style=\"color:blue;\">" +
            cbCalibFiles[i])
    return        


def pbCbCalibFindcorners_clicked(self):
    # This code does not really run. It is only for intellisense 
    # so that editor recognizes the functions and variables of the self. 
    if type(self) == type(None):
        self.ui = Ui_improMeasure()
    # self.ui.pteMsg.appendHtml("<p style=\"color:black;\">Called pbCbCalibFindcorners_clicked()")
    self.uiPrint("Called pbCbCalibFindcorners_clicked", html=True, color='blue')
    # get file lists
    # get the user defined directory of source photos
    cbCalibDir = self.ui.edCbCalibSrcDir.text()
    # if directory does not exist, clear file list and return
    if os.path.exists(cbCalibDir) == False:
        self.ui.pteCbCalibSrcPhotos.clear()
        return
    # list all files
    cbCalibFiles_all = glob.glob(os.path.join(cbCalibDir,'*'))
    # if no file, return
    if type(cbCalibFiles_all) != list or len(cbCalibFiles_all) <= 0 \
        or type(cbCalibFiles_all[0]) != str:
        self.ui.pteMsg.appendHtml("<p style=\"color:red;\">Cannot find calibration file in %s" % cbCalibDir)
        return
    cbCalibFiles = imreadSupportedFiles(cbCalibFiles_all)
    if type(cbCalibFiles) != list or len(cbCalibFiles) <= 0 \
        or type(cbCalibFiles[0]) != str:
        self.ui.pteMsg.appendHtml("<p style=\"color:red;\">Cannot find calibration file in %s" % cbCalibDir)
        return
    # display source files of cbCalib
    nfiles = len(cbCalibFiles)
    self.ui.pteCbCalibSrcPhotos.clear()
    for i in range(nfiles):
        self.ui.pteCbCalibSrcPhotos.appendHtml("<p style=\"color:blue;\">" +
            cbCalibFiles[i])
    # get pattern size (number of corners)
    try:
        strPsizeX = self.ui.edCbCalibPsizeX.text()
        strPsizeY = self.ui.edCbCalibPsizeY.text()
        patternSize = (int(strPsizeX), int(strPsizeY))
    except:
        self.ui.pteMsg.appendHtml("<p style=\"color:red;\">Cannot get pattern size")
        return
    # try to find corners
    imgPoints2f = np.zeros((nfiles * patternSize[0] * patternSize[1],2), 
                           dtype=np.float32)
    imgPoints2f = imgPoints2f.reshape((nfiles,-1,2))
    filenames = []
    for i in range(nfiles):
        filename = os.path.split(cbCalibFiles[i])[1]
        filenames.append(filename)
        # find corners display message
        msgTxt = "Reading %s" % filename
        self.ui.pteMsg.appendHtml("<p style=\"color:black;\">%s" % msgTxt)
        print(msgTxt, flush=True)
        img = cv.imread(cbCalibFiles[i], cv.IMREAD_GRAYSCALE)
        # progress bar
        progress = int((i + 0.1) * 100. / nfiles + 0.5)
        self.ui.progressCbCalib.setValue(progress)
        if (type(img) != type(None) and img.shape[0] > 0 
            and img.shape[1] > 0):
            self.ui.edCbCalibImgWidth.setText("%d" % img.shape[1])
            self.ui.edCbCalibImgHeight.setText("%d" % img.shape[0])
        else:
            self.ui.pteMsg.appendHtml("<p style=\"color:red;\"> Cannot load image from %s" % filename)
            self.ui.pteMsg.ref
            continue
        # display message 
        msgTxt = "Finding corners in %s " % filename
        self.ui.pteMsg.appendHtml("<p style=\"color:black;\">%s" % msgTxt)
        print(msgTxt, flush=True)
        # find corners
#        found, ptsThis = cv.findChessboardCorners(img, patternSize)
        found, ptsThis = cv.findChessboardCornersSB(img, patternSize)
        progress = int((i + 0.8) * 100. / nfiles + 0.5)
        self.ui.progressCbCalib.setValue(progress)
        if found==True:
            msgTxt = "Found sufficient corners"
            print(msgTxt, flush=True)
            self.ui.pteMsg.appendHtml("<p style=\"color:blue;\">%s" % msgTxt)
            imgPoints2f[i,:,:] = ptsThis[:,0,:]
        else:
            msgTxt = "Failed to find sufficient corners"
            print(msgTxt, flush=True)
            self.ui.pteMsg.appendHtml("<p style=\"color:blue;\">%s" % msgTxt)
        progress = int((i + 1.0) * 100. / nfiles + 0.5)
        self.ui.progressCbCalib.setValue(progress)
        # draw corners 
        imgWithCorners = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        cv.drawChessboardCorners(imgWithCorners, patternSize, ptsThis, found)
        fnameRoot, fnameExt = os.path.splitext(cbCalibFiles[i])
        imgCornersFile = fnameRoot + "_cornersDrawn.jpg"
        cv.imwrite(imgCornersFile, imgWithCorners)

    # save corners coordinates to xlsx file (chessboardCalibration.xlsx)
    xlsFile = os.path.join(cbCalibDir, 'chessboardCalibration.xlsx')
      # file list
    columns = ['Photo file name']
    index = []
    for j in range(nfiles):
        index.append('%d' % (j + 1))
    df = pd.DataFrame(filenames, index=index, columns=columns)
      # check if the xls file is new or existed 
    writeDataFrameToExcel(df, xlsFile, 'fileList')
      # corner image points
    columns = []
    index = []
    for j in range(nfiles):
        columns.append('Xcorners_%d' % (j + 1))
        columns.append('Ycorners_%d' % (j + 1))
    for j in range(imgPoints2f.shape[1]):
        index.append('Point %d' % (j + 1))
    df = pd.DataFrame(np.transpose(imgPoints2f, (1,0,2)).reshape(-1, 2 * nfiles), 
         index=index, columns=columns)
    writeDataFrameToExcel(df, xlsFile, 'corners')


def pbCbCalibCalibCam_clicked(self):
    # This code does not really run. It is only for intellisense 
    # so that editor recognizes the functions and variables of the self. 
    if type(self) == type(None):
        self.ui = Ui_improMeasure()
    self.ui.pteMsg.appendHtml("<p style=\"color:blue;\">Called pbCbCalibCalibCam_clicked()")


def pbCbCalibMarkPrjpoints_clicked(self):
    # This code does not really run. It is only for intellisense 
    # so that editor recognizes the functions and variables of the self. 
    if type(self) == type(None):
        self.ui = Ui_improMeasure()
    self.ui.pteMsg.appendHtml("<p style=\"color:blue;\">Called pbCbCalibMarkPrjpoints_clicked()")


def pbCbCalibUndistort_clicked(self):
    # This code does not really run. It is only for intellisense 
    # so that editor recognizes the functions and variables of the self. 
    if type(self) == type(None):
        self.ui = Ui_improMeasure()
    self.ui.pteMsg.appendHtml("<p style=\"color:blue;\">Called pbCbCalibUndistort_clicked()")

