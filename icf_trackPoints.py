# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 13:10:23 2024

@author: yuans
"""

import os
import time
import numpy as np
import glob
#import threading
import cv2 as cv
from math import cos, sin, atan2, pi, sqrt
#from numba import njit

from pickTemplates import pickTemplates
from inputs import input2
from triangulatePoints2 import triangulatePoints2
#from imshow2 import imshow2
#from mgridOnImage import mgridOnImage
#from viewField import viewField
from projectPoints_mp import projectPoints_mp
from calcCrackField import calcCrackField
from calcStrainFields import calcStrainFields


def icf_trackPoints(
    _files=None,
    _ctrlpFile=None,
    _tMethod=None,
    _maxMove=None,
    _wdir=None):
    
    # files
    if type(_files) == type(None):
        print("# Enter file path of image source:")
        print(r"#    E.g., c:\img1\IMG_%04d.tiff 1001 500 ")
        print(r"#      which is, c-style-path start_index num_images")
        print(r"#    E.g., examples\2022rcwall\leftPause\*.JPG")
        _files = input2()
    if _files.find('%') >= 0:
        # c-style form:  path_with_%  start_idx  num_imgs
        files_list = _files.split()
        files_cstyle = files_list[0]
        start_idx = int(files_list[1])
        num_imgs = int(files_list[2])
        files = []
        for i in range(num_imgs):
            files.append(files_cstyle % (start_idx + i))
    else:
        # glob form: containing wildcast symbols
        files = glob.glob(_files)
 
    # poi points on image (ctrlp)
    if type(_ctrlpFile) == type(None):
        print('# Enter file of points of interests (POIs):')
        print("#   E.g., c:/examples/2022rcwall/Sony_poi.txt")
        print("#   The file should be a N-by-6 array.")
        print("#   6 columns are x y x0 y0 width height")
        print("#   x y are image coordinates. x0 y0 width height are the location and size of the template.")
        print('#   If file does not exist, this program asks you to pick by mouse.')
        _ctrlpFile = input2()
    if os.path.isfile(_ctrlpFile):
        # try to load from file 
        # Here we allow delimiters of ',' and ' '
        try:
            with open(_ctrlpFile) as f:
                ctrlp = np.loadtxt((x.replace(',',' ') for x in f))
                npoi = ctrlp.shape[0]
        except:
            pass
    else:
        ctrlp = np.zeros((1,1), dtype=float)
        
    # if file does not exsit or it fails to read data, ask user to pick.
    if ctrlp.shape[1] != 6:
        # pick templates of POIs
        print("# Cannot load POI points from file. Please pick points by mouse.")
        iWaitFile = 0
        while True:
            # wait for file
            if os.path.exists(files[0]) == False:
                print("# Waiting for file %s" % files[0])
                time.sleep(1.0)
                iWaitFile += 1
                continue
            if iWaitFile >= 1:
                time.sleep(1.0)
            break
        img0 = cv.imread(files[0])
        while True:
            print("# How many points do you want to track?")
            try:
                npoi = int(input2())
                break
            except:
                print("# Invalid input. Try again.")   
        ctrlp = pickTemplates(img=img0, nPoints=npoi, savefile=_ctrlpFile,
                              saveImgfile=_ctrlpFile + '.JPG')

    # what method 
    if type(_tMethod) == type(None):
        # ask user to select a tracking method
        while True:
            try:
                print("# Which tracking method do you want to use: ")
                print("#   (1) template match.")
                _tMethod = input2()
                tMethod = int(_tMethod)
                break
            except:
                print("# Invalid input. Try again.")
        
        
        
    # preparation for running loop 
    # (memory allocation)
    nstep = len(files)
    ctrlPoints2d = np.zeros((nstep, npoi, 2), dtype=float)
#    ctrlps = []
#    ctrlps.append(ctrlp)

    # images of the step that templates are defined at 
    tmplt_step = 0
    rectfs = []
    rectfs_tmplt = []
    tic = time.time()
    # wait for file
    iWaitFile = 0
    while True:
        if os.path.exists(files[tmplt_step]) == False:
            print("# Waiting for %s" % files[tmplt_step])
            time.sleep(1.0)
            iWaitFile += 1
            continue
        if iWaitFile >= 1:
            time.sleep(1.0)
        break
    img_tmplt = cv.imread(files[tmplt_step])
    toc = time.time()
    print("# It took %f sec. to read images (Step 0)." % (toc - tic))
    for ipoint in range(npoi):
        ctrlPoints2d[0, ipoint, :] = ctrlp[ipoint, 0:2]

    # estimated maximum displacements 
    # (for estimating size of template-match search region)
    if type(_maxMove) == type(None):
        print("# Enter the estimated maximum movement (along x and y) between photos:")
        print("#   (measured by pixels)")
        print("#   (If you entered 0 0 or -1 -1, they will be estimated by 5 times template size")
        print("#   E.g., 30 20")               
        _maxMove = input2()
    # The actual maxMove is determined after rect_w and rect_h are known.    
    # maxMove = np.fromstring(_maxMove, sep=' ')
    # if maxMove[0] <= 0 or maxMove[1] <= 0:
    #     maxMove[0] = rect_w * 5
    #     maxMove[1] = rect_h * 5


    # get working directory _wdir
    if type(_wdir) == type(None):
        print('# Enter working directory (for output):')
        _wdir = input2()


    # start of the loop
    for istep in range(nstep):
        # wait for file
        iWaitFile = 0
        while True:
            if os.path.exists(files[istep]) == False:
                print("# Waiting for %s" % files[istep])
                time.sleep(1.0)
                iWaitFile += 1
                continue
            if iWaitFile >= 1:
                time.sleep(1.0)
            break
        img = cv.imread(files[istep])
        
        # tracking 
        for ipoint in range(npoi):
            # calculate rect of template (rect_x, rect_y, rect_w, rect_h)
            print("************************************")
            print("Step %d Point %d" % (istep +1, ipoint +1))
            rect_x = round(ctrlp[ipoint][2]\
                           - ctrlp[ipoint][0]\
                           + ctrlPoints2d[tmplt_step, ipoint, 0])
            rect_y = round(ctrlp[ipoint][3]\
                           - ctrlp[ipoint, 1]\
                           + ctrlPoints2d[tmplt_step, ipoint, 1])
            rect_w = int(ctrlp[ipoint, 4])
            rect_h = int(ctrlp[ipoint, 5])
            if rect_x < 0:
                rect_x = 0
            if rect_y < 0:
                rect_y = 0
            if rect_x + rect_w > img.shape[1]:
                rect_x = img.shape[1] - rect_w 
            if rect_y + rect_h > img.shape[0]:
                rect_y = img.shape[0] - rect_h
            # define template
            tmplt = img_tmplt[rect_y:rect_y + rect_h, 
                              rect_x:rect_x + rect_w].copy()
            # The actual maxMove is determined after rect_w and rect_h are known.    
            maxMove = np.fromstring(_maxMove, sep=' ')
            if maxMove[0] <= 0 or maxMove[1] <= 0:
                maxMove[0] = rect_w * 5
                maxMove[1] = rect_h * 5
            srch_x = round(rect_x - maxMove[0])
            srch_y = round(rect_y - maxMove[1])
            srch_w = round(rect_w + 2 * maxMove[0])
            srch_h = round(rect_h + 2 * maxMove[1])
            if srch_x < 0:
                srch_x = 0
            if srch_y < 0:
                srch_y = 0
            if srch_x + srch_w > img.shape[1]:
                srch_x = img.shape[1] - srch_w 
            if srch_y + srch_h > img.shape[0]:
                srch_y = img.shape[0] - srch_h
            # define searched image
            srchd = img[srch_y:srch_y + srch_h, 
                        srch_x:srch_x + srch_w].copy()
            # show
            #print('Searched(step %d). C%d P%d Rect:%d %d %d %d.' %
            #          (istep + 1, icam+1, ipoint+1, 
            #           srch_x, srch_y, srch_w, srch_h))
                # run template match
            tic_tm = time.time()
            tmRes = cv.matchTemplate(srchd, tmplt,\
                                     cv.TM_CCORR_NORMED)
            tmResMinMaxLoc = cv.minMaxLoc(tmRes)
            toc_tm = time.time()
            locxi, locyi = tmResMinMaxLoc[3]
#                print('Template match(step %d). C%d P%d:' %
#                      (istep + 1, icam+1, ipoint+1))
#                print('# It took %f sec. to do template match (step %d, cam %d, point %d).'
#                      % (toc_tm - tic_tm, istep+1, icam+1, ipoint+1))
                # show 
            showTmpltMatched = False
            if showTmpltMatched:
                srchd_show = srchd.copy()
                cv.rectangle(srchd_show, 
                             np.array([locxi, locyi], dtype=np.int32), 
                             np.array([locxi + rect_w, locyi + rect_h], dtype=np.int32),
                             color=(0, 255, 0), thickness=2)
#                cv.imshow('TM:S-%d.C-%d.P-%d', srchd_show)
#                cv.waitKey(0)
                  
            # subpixel adjustment (locxi->locx, locyi->locy)
            if locxi - 1 >= 0 and locxi + 1 < tmRes.shape[1]:
                c0 = tmRes[locyi, locxi - 1]
                c1 = tmRes[locyi, locxi + 0]
                c2 = tmRes[locyi, locxi + 1]
                denom = (c0 - 2 * c1 + c2)
                if abs(denom) > 1e-12:
                    locx = locxi - (-0.5 * c0 + 0.5 * c2) / denom
                else:
                    locx = locxi
            else:
                locx = locxi
            if locyi - 1 >= 0 and locyi + 1 < tmRes.shape[0]:
                c0 = tmRes[locyi - 1, locxi]
                c1 = tmRes[locyi + 0, locxi]
                c2 = tmRes[locyi + 1, locxi]
                denom = (c0 - 2 * c1 + c2)
                if abs(denom) > 1e-12:
                    locy = locyi - (-0.5 * c0 + 0.5 * c2) / denom
                else:
                    locy = locyi
            else:
                locy = locyi
            ctrl_px = locx + srch_x + ctrlp[ipoint][0] - rect_x
            ctrl_py = locy + srch_y + ctrlp[ipoint][1] - rect_y
            # 
            ctrlPoints2d[istep, ipoint, 0] = ctrl_px
            ctrlPoints2d[istep, ipoint, 1] = ctrl_py
        # end of tracking ctrlPoints           
        toc = time.time()
    
    
    
    return
