# This Python file uses the following encoding: utf-8
import os, sys, time

from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog
from PySide6 import QtGui, QtCore
import glob

from createFileList import createFileList
from ui_form import Ui_improMeasure


def pbBasicInfoBrowseRootdir_clicked(self):
    # This code does not really work. It is only for intellisense 
    # so that editor recognizes the functions and variables in self. 
    if type(self) == type(None):
        self.ui = Ui_improMeasure()
    # This function allows user to set root directory by GUI file dialog
    folder = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
    if len(folder) >= 1:
        self.ui.edRootdir.setText(folder)
        # display message
        self.ui.pteMsg.appendPlainText("# (%s) Set %s as the root directory." \
            % (time.strftime("%Y-%m-%d %H:%M:%S"), folder))
    else:
        # display message
        self.ui.pteMsg.appendPlainText("# (%s) Action cancelled. " \
            "Root directory unchanged." % (time.strftime("%Y-%m-%d %H:%M:%S")))

def pbBasicInfoBrowseConfig_clicked(self):
    # This code does not really work. It is only for intellisense 
    # so that editor recognizes the functions and variables in self. 
    if type(self) == type(None):
        self.ui = Ui_improMeasure()
    # This function allows user to set root directory by GUI file dialog
    filePathRet = QFileDialog.getOpenFileName(self, "Select configuration file", )
    filePath = filePathRet[0]
    if len(filePath) >= 1:
        self.ui.edConfig.setText(filePath)
        # display message
        self.ui.pteMsg.appendPlainText("# (%s) Set %s as the configuration file." \
            % (time.strftime("%Y-%m-%d %H:%M:%S"), filePath))
    else:
        # display message
        self.ui.pteMsg.appendPlainText("# (%s) Action cancelled. " \
            "Configuration file unchanged." % (time.strftime("%Y-%m-%d %H:%M:%S")))