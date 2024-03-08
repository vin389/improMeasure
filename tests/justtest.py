import cv2 as cv
import os, glob
import numpy as np
vpath = 'e:\\DCIM\\100MEDIA'
mp4Files = glob.glob(os.path.join(vpath, '*.MP4'))
for i in range(len(mp4Files)):
    vid = cv.VideoCapture(mp4Files[0])
    vid.get(cv.CAP_)
    
    











# import os, sys, time
# import cv2 as cv
# import numpy as np

# from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog
# from PySide6 import QtGui, QtCore
# import glob

# from createFileList import createFileList
# from ui_form import Ui_improMeasure

# print(type(Ui_improMeasure))
# x = Ui_improMeasure()
# type(x)




# import datetime 
# import time
# for i in range(1000):
#     now = datetime.datetime.utcnow()
#     format1 = "%Y-%m-%d %H:%M:%S.%f"
#     print(now.strftime(format1), end='', flush=True)
#     time.sleep(0.05)
#     print('\b' * 30, end='')

