import os
from PySide6 import QtGui, QtCore
from PySide6.QtWidgets import QFileDialog
from ui_form import Ui_improMeasure
import cv2 as cv
import numpy as np
import time

def pbPreviewSelectPics_clicked(self):
     # This code does not really run. It is only for intellisense
     # so that editor recognizes the functions and variables of the self.
     if type(self) == type(None):
         self.ui = Ui_improMeasure()
     # Popup file dialog allowing user to select multiple files for source pictures
     self.preview_srcFiles = QFileDialog.getOpenFileNames(self, "Select Pictures", dir=".")
     nfile = len(self.preview_srcFiles[0])
     if nfile <= 0:
         return
     # Display the source directory and files (assuming all files are in the same directory)
     # The item text includes file name, file index (starting from 1), ctime (created time), and mtime (modified time)
     theDir = os.path.split(self.preview_srcFiles[0][0])[0]
     self.ui.lbPreviewSrcDir.setText(theDir)
     self.ui.cmbPreviewSrcPics.clear()
     for ifile in range(nfile):
         theFilename = os.path.split(self.preview_srcFiles[0][ifile])[1]
         # get ctime and mtime information
         ctime = time.ctime(os.path.getctime(self.preview_srcFiles[0][ifile]))
         mtime = time.ctime(os.path.getmtime(self.preview_srcFiles[0][ifile]))
         itemText = "%s (%d)(%s)(%s)" % (theFilename, ifile + 1, ctime, mtime)
         self.ui.cmbPreviewSrcPics.addItem(itemText)

def cmbPreviewSrcPics_currentIndexChanged(self):
    # This code does not really run. It is only for intellisense 
    # so that editor recognizes the functions and variables of the self. 
    if type(self) == type(None):
        self.ui = Ui_improMeasure()
    try:
        # read the image of the combo file in p2v 
        file = self.preview_srcFiles[0][self.ui.cmbPreviewSrcPics.currentIndex()]
        if os.path.isfile(file) == False:
            print("# Warning: cmbPreviewSrcPics_currentIndexChanged: File does not exist.")
            return
        img1 = cv.imread(file)
        # resize the image to fit the preview label size
        h1, w1, nc = img1.shape
        #   the actual size on the screen could be larger than Qt widget size.
        #   E.g., Windows recommends a scale of 1.25 
        scale = 1.25
        h2, w2 = int(self.ui.lbPreviewImshow.height() * scale), int(self.ui.lbPreviewImshow.width() * scale)
        ratio = min(h2 / h1, w2 / w1)
        dsize = (int(w1 * ratio + .5), int(h1 * ratio + .5))
        img1 = cv.resize(img1, dsize, interpolation=cv.INTER_LANCZOS4)
        # copy img1 (original aspect ratio) to img2 (Widget aspect ratio)
        if nc == 3:
            img2 = np.ones((h2, w2, nc), dtype=np.uint8) * 255
            x0 = int((w2 - img1.shape[1]) / 2)
            y0 = int((h2 - img1.shape[0]) / 2)
            img2[y0:y0+img1.shape[0], x0:x0+img1.shape[1],:] = img1
        else:
            img2 = np.ones((h2, w2), dtype=np.uint8) * 255
            x0 = int((img1.shape[1] - w2) / 2)
            y0 = int((img1.shape[0] - h2) / 2)
            img2[y0:y0+img1.shape[0], x0:x0+img1.shape[1]] = img1
        # create QImage and QPixmap
        bytesPerLine = nc * w2
        qImg = QtGui.QImage(img2.data, w2, h2, bytesPerLine, QtGui.QImage.Format_BGR888)
        pixmap01 = QtGui.QPixmap.fromImage(qImg)
        pixmap_image = QtGui.QPixmap(pixmap01)
        # show QPixmap to Label
        self.ui.lbPreviewImshow.setPixmap(pixmap_image)
        self.ui.lbPreviewImshow.setAlignment(QtCore.Qt.AlignCenter)
        self.ui.lbPreviewImshow.setScaledContents(True)
        self.ui.lbPreviewImshow.setMinimumSize(1,1)
        self.ui.lbPreviewImshow.show()
    except:
        print("# Error: Exception in cmbPreviewSrcPics_currentIndexChanged()")
        return
