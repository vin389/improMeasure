import cv2
import numpy as np
from imshow2 import imshow2


file1 = r'D:\ExpDataSamples\20240600-CarletonShakeTableCeilingSystem\preparation_demo10\Cam 1.mp4'

v1 = cv2.VideoCapture(file1)
if not v1.isOpened(): 
    print("Error opening video file")
    
#ret, imgInit = v1.read()
#if ret == True:
#    imshow2('TEST', imgInit, winmax=(600, 400))


    
    
    
if v1.isOpened():  
    v1.release()    
