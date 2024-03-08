# This Python file uses the following encoding: utf-8
import os, sys, time

from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog
from PySide6 import QtGui, QtCore
import glob

from createFileList import createFileList
from ui_form import Ui_improMeasure

def pbFileListDisplayWildcard_clicked(self):
    """
    This is the event function that is called when the button "Display" 
    in the File List tab named pbFileListDisplayWildcard is pressed by 
    user. 
    This is also the event function that is called when ENTER is pressed 
    in the edit line in the File List tab named edFileListWildcardInput.
    This function:
    reads the user assigned file names with wildcard,
    calls function createFileList() to get the file list,
    displays the files in the plain-text editor (pteFileListDisplay), 
    displays file count and existed file count. 
    """
    # This code does not really work. It is only for intellisense 
    # so that editor recognizes the functions and variables in self. 
    if type(self) == type(None):
        self.ui = Ui_improMeasure()
    # get file list
    wildcardInput = self.ui.edFileListWildcardInput.text()
    files = createFileList(files=wildcardInput, savefile='.')
    # display file list
    if type(files) != list:
        self.ui.pteFileListDisplay.clear()
        self.ui.lbFileListFileCountAll.setText("Number of files: 0")
        self.ui.lbFileListCountExisted.setText("Numver of existed files: 0")
        return
    if len(files) <= 0:
        self.ui.pteFileListDisplay.clear()
        self.ui.lbFileListFileCountAll.setText("Number of files: 0")
        self.ui.lbFileListCountExisted.setText("Numver of existed files: 0")
        return
    nfile = len(files)
    nfileExisted = 0
    self.ui.pteFileListDisplay.clear()
    for i in range(len(files)):
        if os.path.exists(files[i]) == True:
            self.ui.pteFileListDisplay.appendHtml("<p style=\"color:blue;\">%s" % files[i])
            nfileExisted += 1
        else:
            self.ui.pteFileListDisplay.appendHtml("<p style=\"color:red;\">%s" % files[i])
    # display file count
    self.ui.lbFileListFileCountAll.setText("Number of files: %d" % nfile)
    self.ui.lbFileListCountExisted.setText("Numver of existed files: %d (displayed in blue)" % nfileExisted)


def pbFileListDisplayCstyle_clicked(self):
    """
    This is the event function that is called when the button "Display" 
    in the File List tab named pbFileListDisplayCstyle is pressed by 
    user. 
    This is also the event function that is called when ENTER is pressed 
    in the edit line in the File List tab named edFileListCStyleInput, 
    edFileListCStyleStart, and edFileListCStyleCount.
    This function:
    reads the user assigned file names with c-style, its start index, and count,
    calls function createFileList() to get the file list,
    displays the files in the plain-text editor (pteFileListDisplay), 
    displays file count and existed file count. 
    """
    # This code does not really work. It is only for intellisense 
    # so that editor recognizes the functions and variables in self. 
    if type(self) == type(None):
        self.ui = Ui_improMeasure()
    # get file list
    try:
        cstyleInput = self.ui.edFileListCstyleInput.text()
        cstyleStart = int(self.ui.edFileListCstyleStart.text())
        cstyleCount = int(self.ui.edFileListCstyleCount.text())
    except:
        self.ui.pteFileListDisplay.clear()
        self.ui.lbFileListFileCountAll.setText("Number of files: 0")
        self.ui.lbFileListCountExisted.setText("Numver of existed files: 0")
        return
    files = createFileList(files=cstyleInput, cStartIdx=cstyleStart, cNumFiles=cstyleCount, savefile=".")
    # display file list
    if type(files) != list:
        self.ui.pteFileListDisplay.clear()
        self.ui.lbFileListFileCountAll.setText("Number of files: 0")
        self.ui.lbFileListCountExisted.setText("Numver of existed files: 0")
        return
    if len(files) <= 0:
        self.ui.pteFileListDisplay.clear()
        self.ui.lbFileListFileCountAll.setText("Number of files: 0")
        self.ui.lbFileListCountExisted.setText("Numver of existed files: 0")
        return
    nfile = len(files)
    nfileExisted = 0
    self.ui.pteFileListDisplay.clear()
    for i in range(len(files)):
        if os.path.exists(files[i]) == True:
            self.ui.pteFileListDisplay.appendHtml("<p style=\"color:blue;\">%s" % files[i])
            nfileExisted += 1
        else:
            self.ui.pteFileListDisplay.appendHtml("<p style=\"color:red;\">%s" % files[i])
    # display file count
    self.ui.lbFileListFileCountAll.setText("Number of files: %d" % nfile)
    self.ui.lbFileListCountExisted.setText("Numver of existed files: %d (displayed in blue)" % nfileExisted)
