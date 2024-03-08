import os
from PySide6 import QtGui, QtCore
from PySide6.QtWidgets import QFileDialog
from ui_form import Ui_improMeasure
import cv2 as cv
import numpy as np
import time
import glob

from improMisc import inputdlg, trainImgFilesToImgArray, imgGridSlice, slicedMasksToCategorical, showImgMskCateg

def pbMlGridSlice_clicked(self):
    # This code does not really run. It is only for intellisense
    # so that editor recognizes the functions and variables of the self.
    if type(self) == type(None):
        self.ui = Ui_improMeasure()
    # 
    prompt = ['Directory of images', 'Patch size (pixels)', 
              'Directory of masks', 'Output npz file']
    title = 'Images/Masks (*.JPG) to Grid Sliced Training Data (.npz)'
    defaultanswer = [r'D:\yuansen\ImPro\improMeasure\examples\crackTraining\images\*.JPG', 
                     '32',
                     r'D:\yuansen\ImPro\improMeasure\examples\crackTraining\masks\*.JPG',
                     r'D:\yuansen\ImPro\improMeasure\examples\crackTraining\train_patch_32_xy.npz']
    inputs = inputdlg(prompt, title, 1, defaultanswer)
    if type(inputs) == type(None) or len(inputs) <= 0:
        self.uiPrint("pbMlGridSlice_clicked(): inputdlg() is cancelled by user.")
        return
    imgFiles = glob.glob(inputs[0])
    if len(imgFiles) < 1:
        self.uiPrint("pbMlGridSlice_clicked(): Error: "
            "Directory of images (%s) has no file." % (inputs[0]))
        return
    patchSize = int(inputs[1])
    mskFiles = glob.glob(inputs[3])
    if len(mskFiles) < 1:
        self.uiPrint("pbMlGridSlice_clicked(): Warning: "
            "Directory of masks (%s) has no file. Mask processing is skipped." % (inputs[3]))
    imgArray = trainImgFilesToImgArray(imgFiles)
    slicedImgs = imgGridSlice(imgArray, patchSize, patchSize)
    if len(mskFiles) > 0:
        mskArray = trainImgFilesToImgArray(mskFiles)
        slicedMsks = imgGridSlice(mskArray, patchSize, patchSize)
        categ = slicedMasksToCategorical(slicedMsks)
    #
    # showImgMskCateg(slicedImgs[:,:,:,:], slicedMsks[:,:,:,:], categ, (7, 7))
    #
    if len(mskFiles) > 0:
        self.uiPrint("Writing training data (x_train, y_train) to file (%s). Please wait..." % inputs[4])
        xsh = slicedImgs.shape
        ysh = categ.shape
        self.uiPrint("  x_train.shape is (%d,%d,%d,%d)" % (xsh[0], xsh[1], xsh[2], xsh[3]))
        self.uiPoint("  y_train.shape is (%d,%d)" % ysh[0], ysh[0])
        self.ui.progress.setValue(0)
        np.savez_compressed(inputs[4], x_train=slicedImgs, y_train=categ)
    else:
        self.uiPrint("Writing training data (x_train, y_train) to file (%s). Please wait..." % inputs[4])
        xsh = slicedImgs.shape
        ysh = categ.shape
        self.uiPrint("  x_train.shape is (%d,%d,%d,%d)" % (xsh[0], xsh[1], xsh[2], xsh[3]))
        self.uiPoint("  y_train.shape is (%d,%d)" % ysh[0], ysh[0])
        self.ui.progress.setValue(0)
        np.savez_compressed(inputs[4], x_train=slicedImgs, y_train=categ)

    self.ui.pteMsg.appendPlainText("# Writing completed.")
    self.ui.progress.setValue(100)    
    print("# Writing completed.")


