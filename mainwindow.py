
# This Python file uses the following encoding: utf-8
import os, sys, time, datetime

from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog

# Important:
# You need to run the following command to generate the ui_form.py file
#     pyside6-uic form.ui -o ui_form.py, or
#     pyside2-uic form.ui -o ui_form.py
from ui_form import Ui_improMeasure
import cv2 as cv
import numpy as np

from improUiBasicInfo import pbBasicInfoBrowseRootdir_clicked, pbBasicInfoBrowseConfig_clicked
from improUiPreview import pbPreviewSelectPics_clicked, cmbPreviewSrcPics_currentIndexChanged
from improUiCbCalib import cmbCbCalibCam_currentIndexChanged, edCbCalibSrcDir_returnPressed,\
     pbCbCalibFindcorners_clicked, pbCbCalibCalibCam_clicked,\
     pbCbCalibMarkPrjpoints_clicked, pbCbCalibUndistort_clicked
from improUiFileList import pbFileListDisplayWildcard_clicked, pbFileListDisplayCstyle_clicked
from improUiMlUtils import pbMlGridSlice_clicked
#from improUiPoints import pbPickPoints_clicked
from createFileList import ufilesToFileList
from improOpticalFlow import demoOpticalFlow
from improTemplateMatch import demoTemplateMatch
from improEcc import demoEcc
from improFeatureMatch import demoFeatureMatch
from improMisc import demoVideoFrameCount

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_improMeasure()
        self.ui.setupUi(self)
        # event binding
        #   in Basic info tab
        theCD = os.getcwd()
        self.ui.edCurrentDirectory.setText(theCD)
        self.ui.pbBrowseRootdir.clicked.connect(lambda: pbBasicInfoBrowseRootdir_clicked)
        self.ui.pbBrowseConfig.clicked.connect(lambda: pbBasicInfoBrowseConfig_clicked)
        self.ui.pbClearMsg.clicked.connect(lambda: self.ui.pteMsg.clear())
        #   in Chessboard Calib (CbCalib)
        self.ui.cmbCbCalibCam.currentIndexChanged.connect(lambda: cmbCbCalibCam_currentIndexChanged)
        self.ui.edCbCalibSrcDir.returnPressed.connect(lambda: edCbCalibSrcDir_returnPressed)
        self.ui.pbCbCalibFindcorners.clicked.connect(lambda: pbCbCalibFindcorners_clicked(self))
        self.ui.pbCbCalibCalibCam.clicked.connect(lambda: pbCbCalibCalibCam_clicked(self))
        self.ui.pbCbCalibMarkPrjpoints.clicked.connect(lambda: pbCbCalibMarkPrjpoints_clicked)
        self.ui.pbCbCalibUndistort.clicked.connect(lambda: pbCbCalibUndistort_clicked)
        #   in ML utils
        self.ui.pbMlGridSlice.clicked.connect(lambda: pbMlGridSlice_clicked)
        #   in SiCalib
#        self.ui.pbSiCalibRead.clicked.connect(self.pbSiCalibRead_clicked)
        #      Push button [Calibrate camera]
        #        1. Read calibration points from configuration xlsx file
        #        2. Read calibration flags from self.edSiCalibFlags
        #        3. Run the single image calibration 
        #        4. Calculate projection points and projection errors
        #        5. Display the camera parameters on widgets
        #        6. Write the calibration result to files
        #        7. Read background image
        #        8. Plot calibration points and projection points on the image
        #        9. Display the image on self.lbSiCalibImg
        #       10. Write the projection image to file (file name: ***_projected.***, where ***.*** is the background image)
#        self.ui.pbSiCalibCalib.clicked.connect(self.pbSiCalibCalib_clicked)
        # Set max number of cameras
        #self.maxNumCam = 6
        # Initialize chessboard calibratino photo file list
        # self.cboardFiles[iCam][iFile] is the relative 
        #self.cboardFiles=[]
        #for i in range(self.maxNumCam):
        #    self.cboardFiles.append([])

        # Images:
        self.ui.pbImagesRefreshList.clicked.connect(lambda: pbImagesRefreshList_clicked(self))
        def pbImagesRefreshList_clicked(self: MainWindow):
            ufiles = self.ui.edImagesFiles.text()
            files = ufilesToFileList(ufiles)
            self.ui.lbImagesList.setText('List of Files (%d files)' % len(files))
            self.ui.pteImagesList.clear()
            for f in files:
                self.ui.pteImagesList.appendPlainText(f)

        # Points:
#        self.ui.pbPickPoints.clicked.connect(lambda: pbPickPoints_clicked(self))

        # Utils: p2v
        self.ui.pbPreviewSelectPics.clicked.connect(lambda: pbPreviewSelectPics_clicked(self))
        self.ui.cmbPreviewSrcPics.currentIndexChanged.connect(lambda: cmbPreviewSrcPics_currentIndexChanged(self))

        # Utilities: 
        self.ui.pbFileListDisplayWildcard.clicked.connect(lambda: pbFileListDisplayWildcard_clicked(self))
        self.ui.edFileListWildcardInput.returnPressed.connect(lambda: pbFileListDisplayWildcard_clicked(self))
        self.ui.pbFileListDisplayCstyle.clicked.connect(lambda: pbFileListDisplayCstyle_clicked(self))
        self.ui.edFileListCstyleInput.returnPressed.connect(lambda: pbFileListDisplayCstyle_clicked(self))
        self.ui.edFileListCstyleStart.returnPressed.connect(lambda: pbFileListDisplayCstyle_clicked(self))
        self.ui.edFileListCstyleCount.returnPressed.connect(lambda: pbFileListDisplayCstyle_clicked(self))
        # push button Optical flow
        self.ui.pbOpticalFlow.clicked.connect(lambda: pbOpticalFlow_clicked(self))
        def pbOpticalFlow_clicked(self):
            demoOpticalFlow()
        # push button template match
        self.ui.pbTemplateMatch.clicked.connect(lambda: pbTemplateMatch_clicked(self))
        def pbTemplateMatch_clicked(self):
            demoTemplateMatch()
        # push button ecc
        self.ui.pbEcc.clicked.connect(lambda: pbEcc_clicked(self))
        def pbEcc_clicked(self):
            demoEcc()
        # push button feature match
        self.ui.pbFeatureMatch.clicked.connect(lambda: pbFeatureMatch_clicked(self))
        def pbFeatureMatch_clicked(self):
            demoFeatureMatch()
        # push button video frame count
        self.ui.pbVideoFrameCount.clicked.connect(lambda: pbVideoFrameCount_clicked(self))
        def pbVideoFrameCount_clicked(self):
            demoVideoFrameCount()


    def uiPrint(self, str, html=False, color='black'):
        nowStr = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-4]
        dispStr = "(%s) %s" % (nowStr, str)
        if html == False:
            self.ui.pteMsg.appendPlainText(dispStr)
        else:
            self.ui.pteMsg.appendHtml("<p style=\"color:%s;\">%s" % (color, dispStr))

    


if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = MainWindow()
    widget.show()
    sys.exit(app.exec())
