# This Python file uses the following encoding: utf-8
import os, sys, time
import cv2 as cv
import numpy as np

from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog
from PySide6 import QtGui, QtCore
import glob

from createFileList import createFileList
from ui_form import Ui_improMeasure
from improMisc import *


def pbPickPoints_clicked(self, x):
    # This code does not really run. It is only for intellisense 
    # so that editor recognizes the functions and variables of the self. 
    if type(self) == type(None):
        self.ui = Ui_improMeasure()
    # 
    self.uiPrint("pbPickPoints_clicked Here!!")
    print(type(x))
    pass
