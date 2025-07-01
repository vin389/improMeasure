import os
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from inputs import *


def drawTemplates(img=[], pts=[], color=[0,255,0], 
                  markerType=cv.MARKER_CROSS, markerSize=20, 
                  thickness=2, lineType=8, 
                  savefile=""):
    # image
    if (img == []):
        img = imread2()
        print("# Image size: %d x %d (width x height)"
              % (img.shape[1], img.shape[0]))
    # pts (xi yi x0 y0 w h)
    if (pts == []):
        pts = readPoints()
        print("# Read %d points (%dx%d)" % 
              (pts.shape[0], pts.shape[0], pts.shape[1]))
    # savefile
    if (savefile == ""):
        print("# Enter the image file to save:")
        print("#  or enter a single character to skip file saving.")
        savefile = input2()
    # draw templates one by one
    pts2 = pts.reshape(-1, pts.shape[-1])
    nPts = pts2.shape[0]
    for i in range(nPts):
        thisPt = pts2[i]
        thisPtInt = [int(thisPt[0]+0.5), int(thisPt[1]+0.5)]
        # draw marker
        img = cv.drawMarker(img,
                      thisPtInt,
                      color=color, 
                      markerType=markerType, 
                      markerSize=markerSize, 
                      thickness=thickness, line_type=lineType)
        # draw box of template
        poly = np.array([pts2[i,2]            , pts2[i,3], 
                         pts2[i,2]            , pts2[i,3] + pts2[i,5], 
                         pts2[i,2] + pts2[i,4], pts2[i,3] + pts2[i,5], 
                         pts2[i,2] + pts2[i,4], pts2[i,3]])
        poly = np.array(poly + 0.5, dtype=int).reshape(1,-1,2)
        isClosed = True
        color = [64, 255, 64]
        thickness = 2
        lineType = cv.LINE_8
        img = cv.polylines(img, poly, isClosed, color, thickness, lineType)
        # draw text
        fontScale = 2.0
        cv.putText(img, "%d" % (i + 1), 
                   thisPtInt,
                   cv.FONT_HERSHEY_PLAIN, 
                   fontScale, color)
    # save 
    if (len(savefile) > 1):
        cv.imwrite(savefile, img)
        print("# Image with markers saved to %s" % (savefile))
    return img

