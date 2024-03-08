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
from icf_drawField import drawField
from projectPoints_mp import projectPoints_mp
from calcCrackField import calcCrackField
from calcStrainFields import calcStrainFields


def icf_wallMonitor_v2(
    _files1=None,
    _files2=None,
    _cam1=None,
    _cam2=None,
    _ctrlp1file=None,
    _ctrlp2file=None,
    _maxMove=None,
    _ppx=None,
    _xzzx=None,
    _rexp=None,
    _cellSizes=None,
    _wdir=None):
    #

    # files1 
    if type(_files1) == type(None):
        print("# Enter file path of camera 1 image source:")
        print(r"#    E.g., c:\img1\IMG_%04d.tiff 1001 500 ")
        print(r"#      which is, c-style-path start_index num_images")
        print(r"#    E.g., examples\2022rcwall\leftPause\*.JPG")
        _files1 = input2()
    if _files1.find('%') >= 0:
        # c-style form:  path_with_%  start_idx  num_imgs
        files1_list = _files1.split()
        files1_cstyle = files1_list[0]
        start_idx_1 = int(files1_list[1])
        num_imgs_1 = int(files1_list[2])
        files1 = []
        for i in range(num_imgs_1):
            files1.append(files1_cstyle % (start_idx_1 + i))
    else:
        # glob form: containing wildcast symbols
        files1 = glob.glob(_files1)
        
    # files2
    if type(_files2) == type(None):
        print("# Enter file path of camera 2 image source:")
        print(r"#    E.g., c:\img2\IMG_%04d.tiff 1001 500 ")
        print(r"#      which is, c-style-path start_index num_images")
        print(r"#    E.g., examples\2022rcwall\rightPause\*.JPG")
        _files2 = input2()
    if _files2.find('%') >= 0:
        # c-style form:  path_with_%  start_idx  num_imgs
        files2_list = _files2.split()
        files2_cstyle = files2_list[0]
        start_idx_2 = int(files2_list[1])
        num_imgs_2 = int(files2_list[2])
        files2 = []
        for i in range(num_imgs_2):
            files2.append(files2_cstyle % (start_idx_2 + i))
    else:
        # glob form: containing wildcast symbols
        files2 = glob.glob(_files2)
        
    # camera parameters file --> imgSize1 cmat1 dvec1 rvec1 tvec1 r441 
    if type(_cam1) == type(None):
        print("# Enter file path of camera 1 parameters:")
        print("#   The file must be in one-column text file, consisting of:")
        print("#     image size (width height, 2 integers),")
        print("#     extrinsic parameters (6 reals),")
        print("#     intrinsic parameters (fx fy cx cy k1 k2 p1 p2 k3 ...) (at least 4 reals)")
        print("#     (See OpenCV manual about camera calibration for details.")
        _cam1 = input2()
    # wait and read file (_cam1)
    while True:
        if os.path.exists(_cam1) == False:
            print("# Waiting for %s" % _cam1)
            time.sleep(1.0)
            continue
        time.sleep(1.0)
        cam1np = np.loadtxt(_cam1)
        break
    imgSize1 = cam1np[0:2].reshape(-1).astype(int)
    rvec1 = cam1np[2:5].reshape(3,1)
    tvec1 = cam1np[5:8].reshape(3,1)
    cmat1 = cam1np[8:17].reshape(3,3)
    dvec1 = cam1np[17:].reshape(-1,1)
    r441 = np.eye(4, dtype=float)
    r441[0:3,0:3] = cv.Rodrigues(rvec1)[0]
    r441[0:3,3] = tvec1.reshape(-1)
        
    # camera parameters file --> imgSize2 cmat2 dvec2 rvec2 tvec2 r442 
    if type(_cam2) == type(None):
        print("# Enter (text) file path of camera 2 parameters:")
        _cam2 = input2()
    # wait and read file (_cam2)
    while True:
        if os.path.exists(_cam2) == False:
            print("# Waiting for %s" % _cam2)
            time.sleep(1.0)
            continue
        time.sleep(1.0)
        cam2np = np.loadtxt(_cam2)
        break
    imgSize2 = cam2np[0:2].reshape(-1).astype(int)
    rvec2 = cam2np[2:5].reshape(3,1)
    tvec2 = cam2np[5:8].reshape(3,1)
    cmat2 = cam2np[8:17].reshape(3,3)
    dvec2 = cam2np[17:].reshape(-1,1)
    r442 = np.eye(4, dtype=float)
    r442[0:3,0:3] = cv.Rodrigues(rvec2)[0]
    r442[0:3,3] = tvec2.reshape(-1)
    
    rvecs = [rvec1, rvec2]
    tvecs = [tvec1, tvec2]
    cmats = [cmat1, cmat2]
    dvecs = [dvec1, dvec2]
    r44s = [r441, r442]
    imgSizes = [imgSize1, imgSize2]
    
    # ctrl points on image of camera 1 (ctrlp1)
    if type(_ctrlp1file) == type(None):
        print('# Enter file of camera 1 ctrl points.')
        print("#   E.g., c:/examples/2022rcwall/Sony_L_ctrlp.txt")
        print("#   The file should be a 3-by-6 array.")
        print("#   6 columns are x y x0 y0 width height")
        print("#   x y are image coordinates. x0 y0 width height are the location and size of the template.")
        print('#   If file does not exist, this program asks you to pick by mouse.')
        _ctrlp1file = input2()
    if os.path.isfile(_ctrlp1file):
        # try to load from file 
        # Here we allow delimiters of ',' and ' '
        try:
            with open(_ctrlp1file) as f:
                ctrlp1 = np.loadtxt((x.replace(',',' ') for x in f))
        except:
            pass
    else:
        ctrlp1 = np.array(0)
    # if file does not exsit or it fails to read data, ask user to pick.
    if ctrlp1.shape != (3,6):
        # pick 3 templates: lower-left (origin), upper-left, lower-right
        print("# Cannot load camera 1 ctrl points from file. Please pick points by mosue.")
        while True:
            # wait for file
            if os.path.exists(files1[0]) == False:
                print("# Waiting for file %s" % files1[0])
                time.sleep(1.0)
                continue
            time.sleep(1.0)
            break
        img0 = cv.imread(files1[0])
        ctrlp1 = pickTemplates(img=img0, nPoints=3, savefile=_ctrlp1file,
                               saveImgfile=_ctrlp1file + '.JPG')

    # ctrl points on image of camera 2 (ctrlp2)
    if type(_ctrlp2file) == type(None):
        print('# Enter file of camera 2 ctrl points.')
        print("#   E.g., c:/examples/2022rcwall/Sony_R_ctrlp.txt")
        print("#   The file should be a 3-by-6 array.")
        print("#   6 columns are x y x0 y0 width height")
        print("#   x y are image coordinates. x0 y0 width height are the location and size of the template.")
        print('#   If file does not exist, this program asks you to pick by mouse.')
        _ctrlp2file = input2()
    if os.path.isfile(_ctrlp2file):
        # try to load from file 
        # Here we allow delimiters of ',' and ' '
        try:
            with open(_ctrlp2file) as f:
                ctrlp2 = np.loadtxt((x.replace(',',' ') for x in f))
        except:
            pass
    else:
        ctrlp2 = np.array(0)
    # if file does not exsit or it fails to read data, ask user to pick.
    if ctrlp2.shape != (3,6):
        # pick 3 templates: lower-left (origin), upper-left, lower-right
        print("# Cannot load camera 2 ctrl points from file. Please pick points by mosue.")
        while True:
            # wait for file
            if os.path.exists(files2[0]) == False:
                print("# Waiting for file %s" % files2[0])
                time.sleep(1.0)
                continue
            time.sleep(1.0)
            break
        img0 = cv.imread(files2[0])
        ctrlp2 = pickTemplates(img=img0, nPoints=3, savefile=_ctrlp2file,
                               saveImgfile=_ctrlp2file + '.JPG')

    # preparation for running loop 
    # (memory allocation)
    nstep = len(files1)
    ctrlPoints2d = np.zeros((nstep, 2, 3, 2), dtype=float)
#    ctrlPoints3d = np.zeros((nstep, 3, 3), dtype=float)
    ctrlps = []
    ctrlps.append(ctrlp1)
    ctrlps.append(ctrlp2)

    # images of the step that templates are defined at 
    tmplt_step = 0
    imgs = []
    imgs_tmplt = []
    rectfs = []
    rectfs_tmplt = []
    tic = time.time()
    # wait for file
    while True:
        if os.path.exists(files1[tmplt_step]) == False:
            print("# Waiting for %s" % files1[tmplt_step])
            time.sleep(1.0)
            continue
        time.sleep(1.0)
        break
    imgs_tmplt.append(cv.imread(files1[tmplt_step]))
    # wait for file
    while True:
        if os.path.exists(files2[tmplt_step]) == False:
            print("# Waiting for %s" % files2[tmplt_step])
            time.sleep(1.0)
            continue
        time.sleep(1.0)
        break
    imgs_tmplt.append(cv.imread(files2[tmplt_step]))
    toc = time.time()
    print("# It took %f sec. to read images (Step 0)." % (toc - tic))
    for icam in range(2):
        for ipoint in range(3):
            ctrlPoints2d[0, icam, ipoint, :] = ctrlps[icam][ipoint, 0:2]

    # start of the loop
    for istep in range(nstep):
    
        # read images
        tic = time.time()
        imgs = []
        # wait for file
        while True:
            if os.path.exists(files1[istep]) == False:
                print("# Waiting for %s" % files1[istep])
                time.sleep(1.0)
                continue
            time.sleep(1.0)
            break
        imgs.append(cv.imread(files1[istep]))
        # wait for file
        while True:
            if os.path.exists(files2[istep]) == False:
                print("# Waiting for %s" % files2[istep])
                time.sleep(1.0)
                continue
            time.sleep(1.0)
            break
        imgs.append(cv.imread(files2[istep]))
        toc = time.time()
        print('# It took took %f sec. to read images (step %d).' % (toc - tic,
              istep + 1))
    
        # tracking/positioning
        if istep == -1:
            # initial step. No tracking is needed.
            for icam in range(2):
                for ipoint in range(3):
                    ctrlPoints2d[istep, icam, ipoint, :] = ctrlps[icam][ipoint, 0:2]
        else:
            # tracking
            for icam in range(2):
                for ipoint in range(3):
                    # calculate rect of template (rect_x, rect_y, rect_w, rect_h)
                    print("************************************")
                    print("Step %d Cam %d Point %d" % (istep +1, icam +1, ipoint +1))
                    rect_x = round(ctrlps[icam][ipoint, 2]\
                           - ctrlps[icam][ipoint, 0]\
                           + ctrlPoints2d[tmplt_step, icam, ipoint, 0])
                    rect_y = round(ctrlps[icam][ipoint, 3]\
                           - ctrlps[icam][ipoint, 1]\
                           + ctrlPoints2d[tmplt_step, icam, ipoint, 1])
                    rect_w = int(ctrlps[icam][ipoint, 4])
                    rect_h = int(ctrlps[icam][ipoint, 5])
                    if rect_x < 0:
                        rect_x = 0
                    if rect_y < 0:
                        rect_y = 0
                    if rect_x + rect_w > imgs[icam].shape[1]:
                        rect_x = imgs[icam].shape[1] - rect_w 
                    if rect_y + rect_h > imgs[icam].shape[0]:
                        rect_y = imgs[icam].shape[0] - rect_h
                    # define template
                    tmplt = imgs_tmplt[icam][rect_y:rect_y + rect_h, 
                                       rect_x:rect_x + rect_w].copy()
                    # show
                    print('Tmplt(step %d). C%d P%d Rect:%d %d %d %d.' %
                          (tmplt_step + 1, icam+1, ipoint+1, 
                           rect_x, rect_y, rect_w, rect_h))
#                    cv.imshow('tmplt', tmplt)
#                    cv.waitKey(10)
#                    cv.destroyWindow('tmplt')

                    # estimated maximum displacements 
                    # (for estimating size of template-match search region)
                    if istep == 0:
                        if type(_maxMove) == type(None):
                            print("# Enter the estimated maximum movement (along x and y) between photos:")
                            print("#   (measured by pixels)")
                            print("#   (If you entered 0 0 or -1 -1, they will be estimated by 5 times template size")
                            print("#   E.g., 30 20")               
                            _maxMove = input2()
                        maxMove = np.fromstring(_maxMove, sep=' ')
                        if maxMove[0] <= 0 or maxMove[1] <= 0:
                            maxMove[0] = rect_w * 5
                            maxMove[1] = rect_h * 5

                    # estimate rect of search image (srch_x, srch_y, srch_w, srch_h)
                    # if istep == 0:
                    #     est_rect_x = rect_x
                    #     est_rect_y = rect_y
                    # elif istep == 1:
                    #     est_rect_x = rect_x\
                    #                + ctrlPoints2d[istep - 1, icam, ipoint, 0]\
                    #                - ctrlPoints2d[        0, icam, ipoint, 0]\
                    #     est_rect_y = rect_y\
                    #                + ctrlPoints2d[istep - 1, icam, ipoint, 1]\
                    #                - ctrlPoints2d[        0, icam, ipoint, 1]\
                    # else:
                    #     est_rect_x = rect_x\
                    #                + ctrlPoints2d[istep - 1, icam, ipoint, 0]\
                    #                - ctrlPoints2d[        0, icam, ipoint, 0]\
                    #                + ctrlPoints2d[istep - 1, icam, ipoint, 0]
                    #                - ctrlPoints2d[istep - 2, icam, ipoint, 0]
                    #     est_rect_y = rect_y\
                    #                + ctrlPoints2d[istep - 1, icam, ipoint, 1]\
                    #                - ctrlPoints2d[        0, icam, ipoint, 1]\
                    #                + ctrlPoints2d[istep - 1, icam, ipoint, 1]
                    #                - ctrlPoints2d[istep - 2, icam, ipoint, 1]
                    srch_x = round(rect_x - maxMove[0])
                    srch_y = round(rect_y - maxMove[1])
                    srch_w = round(rect_w + 2 * maxMove[0])
                    srch_h = round(rect_h + 2 * maxMove[1])
                    if srch_x < 0:
                        srch_x = 0
                    if srch_y < 0:
                        srch_y = 0
                    if srch_x + srch_w > imgs[icam].shape[1]:
                        srch_x = imgs[icam].shape[1] - srch_w 
                    if srch_y + srch_h > imgs[icam].shape[0]:
                        srch_y = imgs[icam].shape[0] - srch_h
                    # define searched image
                    srchd = imgs[icam][srch_y:srch_y + srch_h, 
                                       srch_x:srch_x + srch_w].copy()
                    toc = time.time()
                    # show
                    print('Searched(step %d). C%d P%d Rect:%d %d %d %d.' %
                          (istep + 1, icam+1, ipoint+1, 
                           srch_x, srch_y, srch_w, srch_h))
#                    cv.imshow('searched', srchd)
#                    cv.waitKey(1000)
#                    cv.destroyWindow('searched')
                    # run template match
                    tic = time.time()
                    tmRes = cv.matchTemplate(srchd, tmplt,\
                                             cv.TM_CCORR_NORMED)
                    tmResMinMaxLoc = cv.minMaxLoc(tmRes)
                    locxi, locyi = tmResMinMaxLoc[3]
                    toc = time.time()
#                    print('Template match(step %d). C%d P%d:' %
#                          (istep + 1, icam+1, ipoint+1))
                    print('# It took took %f sec. to do template match (step %d, cam %d, point %d).'
                          % (toc - tic, istep+1, icam+1, ipoint+1))
                    # show 
                    showTmpltMatched = False
                    if showTmpltMatched:
                        srchd_show = srchd.copy()
                        cv.rectangle(srchd_show, 
                                     np.array([locxi, locyi], dtype=np.int32), 
                                     np.array([locxi + rect_w, locyi + rect_h], dtype=np.int32),
                                     color=(0, 255, 0), thickness=2)
#                    cv.imshow('TM:S-%d.C-%d.P-%d', srchd_show)
#                    cv.waitKey(0)
                    
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
                    ctrl_px = locx + srch_x + ctrlps[icam][ipoint][0] - rect_x
                    ctrl_py = locy + srch_y + ctrlps[icam][ipoint][1] - rect_y
                    # 
                    ctrlPoints2d[istep, icam, ipoint, 0] = ctrl_px
                    ctrlPoints2d[istep, icam, ipoint, 1] = ctrl_py
                    
        # triangulation of 3 ctrl points
        imgPoints1 = ctrlPoints2d[istep, 0, 0:3]
        imgPoints2 = ctrlPoints2d[istep, 1, 0:3]
        objPoints, objPoints1, objPoints2, \
            prjPoints1, prjPoints2, prjErrors1, prjErrors2 = \
            triangulatePoints2(cmat1, dvec1, rvec1, tvec1, 
                               cmat2, dvec2, rvec2, tvec2, imgPoints1, imgPoints2)
        print('# Triangulated P1: %12.4e %12.4e %12.4e' % (objPoints[0,0], objPoints[0,1], objPoints[0,2]))
        print('# Triangulated P2: %12.4e %12.4e %12.4e' % (objPoints[1,0], objPoints[1,1], objPoints[1,2]))
        print('# Triangulated P3: %12.4e %12.4e %12.4e' % (objPoints[2,0], objPoints[2,1], objPoints[2,2]))
        print("# Proj. err. P1 (cam1x cam1y cam2x cam2y)(pixels): %.2f %.2f %.2f %.2f"
              % (prjErrors1[0,0], prjErrors1[0,1], prjErrors2[0,0], prjErrors2[0,1]))
        print("# Proj. err. P2 (cam1x cam1y cam2x cam2y)(pixels): %.2f %.2f %.2f %.2f"
              % (prjErrors1[1,0], prjErrors1[1,1], prjErrors2[1,0], prjErrors2[1,1]))
        print("# Proj. err. P3 (cam1x cam1y cam2x cam2y)(pixels): %.2f %.2f %.2f %.2f"
              % (prjErrors1[2,0], prjErrors1[2,1], prjErrors2[2,0], prjErrors2[2,1]))
    
        # Estimate the ppx (pixels per unit length) of original photos
        # (ppx_ori_cam1_x, ppx_ori_cam1_z, ppx_ori_cam2_x, ppx_ori_cam2_z)
        # The ppx do not change over time (istep), so it is estimated only at 
        # istep of 0. 
        if istep == 0:
            dx_pixel_cam1 = np.linalg.norm(ctrlp1[0,0:2] - ctrlp1[2,0:2])
            dz_pixel_cam1 = np.linalg.norm(ctrlp1[0,0:2] - ctrlp1[1,0:2])
            dx_pixel_cam2 = np.linalg.norm(ctrlp2[0,0:2] - ctrlp2[2,0:2])
            dz_pixel_cam2 = np.linalg.norm(ctrlp2[0,0:2] - ctrlp2[1,0:2])
            dx_dist = np.linalg.norm(objPoints[0,:] - objPoints[2,:])
            dz_dist = np.linalg.norm(objPoints[0,:] - objPoints[1,:])
            ppx_ori_cam1_x = dx_pixel_cam1 / dx_dist
            ppx_ori_cam1_z = dz_pixel_cam1 / dz_dist
            ppx_ori_cam2_x = dx_pixel_cam2 / dx_dist
            ppx_ori_cam2_z = dz_pixel_cam2 / dz_dist
            ppx_default = max(ppx_ori_cam1_x, ppx_ori_cam1_z, ppx_ori_cam2_x, ppx_ori_cam2_z) * 2
            print("# Estimated pixels per unit length (P1-P3) in camera 1: %f px/length" % ppx_ori_cam1_x)
            print("# Estimated pixels per unit length (P1-P2) in camera 1: %f px/length" % ppx_ori_cam1_z)
            print("# Estimated pixels per unit length (P1-P3) in camera 2: %f px/length" % ppx_ori_cam2_x)
            print("# Estimated pixels per unit length (P1-P2) in camera 2: %f px/length" % ppx_ori_cam2_z)
    
            # ppx (pixels per unit length for metric rectification)
            if type(_ppx) == type(None):
                print("# Enter how many pixels per unit length (ppx) you want to generate the metric rectification images:")
                print("#   Unit length could be mm, m, inch, or ft, depending on which unit of length is used for camera calibration.")        
                print("#   The greater this value of ppx is, the larger the rectified images are.")
                print("#   The default value (if you input 0 or -1) is %f (twice of estimated density of original photos):"
                      % ppx_default)
                print("#   E.g., -1 (for default value)")
                _ppx = input2()
            ppx = float(_ppx)
            if ppx <= 0.0:
                ppx = ppx_default
            print("#   Pixels per unit length (ppx) is %f." % ppx)
            print("#   Or each pixel is %f unit length." % (1. / ppx))

        # define surface coordinate
        if istep == 0:
            if type(_xzzx) == type(None):
                print("# How do you define surface coord?")
                print("#   E.g.: 0 or xz (surface x is P1 to P3, z is close to P1 to P2.")
                print("#   E.g.: 1 or zx (surface z is P1 to P2, x is close to P1 to P3.")
                _xzzx = input2()
            xzzx = -1
            if _xzzx == '0' or _xzzx == 'xz':
                xzzx = 0
            if _xzzx == '1' or _xzzx == 'zx':
                xzzx = 1

        # initial estimates of surface coordinates (vecx, vecy, and vecz)
        vecx = objPoints[2,:] - objPoints[0,:]
        vecz = objPoints[1,:] - objPoints[0,:]
        vecx = vecx / np.linalg.norm(vecx)
        vecz = vecz / np.linalg.norm(vecz)
        vecy = np.cross(vecz, vecx)
        vecy = vecy / np.linalg.norm(vecy)
        angle = np.math.acos(np.dot(vecx, vecz)) * 180. / np.pi
        print("# Angle between P1->P3 and P1->P2 is %.2f deg." % angle)

        # adjustment of vecx or vecz 
        if xzzx == 0:
            # surface coordinate vecx is P1->P3. Adjust vecz.
            vecz = np.cross(vecx, vecy)
            vecx = vecx / np.linalg.norm(vecx)
            vecy = vecy / np.linalg.norm(vecy)
            vecz = vecz / np.linalg.norm(vecz)
            mc = np.zeros((3,3),dtype=float)
            mc[0,:] = vecx
            mc[1,:] = vecy
            mc[2,:] = vecz
            merr = np.linalg.inv(mc) - mc.transpose() 
            merr = np.max(np.abs(merr.flatten()))
            if merr > 1e-9:
                print("# Error: Surface coordinate is not orthonormal!")
        else: # 
            # surface coordinate vecz is P1->P2. Adjust vecx. 
            vecx = np.cross(vecy, vecz)
            vecx = vecx / np.linalg.norm(vecx)
            vecy = vecy / np.linalg.norm(vecy)
            vecz = vecz / np.linalg.norm(vecz)
            mc = np.zeros((3,3),dtype=float)
            mc[0,:] = vecx
            mc[1,:] = vecy
            mc[2,:] = vecz
            merr = np.linalg.inv(mc) - mc.transpose() 
            merr = np.max(np.abs(merr.flatten()))
            if merr > 1e-9:
                print("# Error: Surface coordinate is not orthonormal!")
    
        # ratio of ROI expansion for metric rectification that user assigns 
        if istep == 0:
            if type(_rexp) == type(None):
                print('# Enter ratio of ROI expansion: ')
                print('#   ROI1 is a tight rectangular region defined by P1-P2-P3.')
                print('#   ROI2 is an expanded region for metric rectification.')
                print('#   E.g., 0.1  (indicating 10% of expansion along 4 sides)')
                print('#               making ROI2 is 20% wider and higher')
                print('#               and ROI2 area is 1.44 times ROI1')
                _rexp = input2()
            rexp = float(_rexp)
            if rexp < 0.0:
                print("# You entered an invalid rexp. It is reset to a default value.")
                rexp = 0.1
            print("# The ratio of expansion is %f" % rexp)
 
        # define roi1 and roi2 (in 3D)
        # roi1 is a rectangular region defined by P1 P2 and P3. 
        # roi2 is the expansion of roi1 with an expansion ration rexp
        # E.g., if rexp is 0.1, 10% of expansion along 4 sides, that is, 
        # the width and height of roi2 are 20% larger than roi1
        # and the area of roi2 is 1.44 times roi1. 
        # roi1 and roi2 are defined by four 3D points stored in a 4-by-3 array.
        # Note: Because numbers of pixels of rectified images must be integers, 
        #       the actual expansion ratio could be slightly adjusted to fit the 
        #       integer numbers of pixels. The actual roi2 is calculated so that 
        #       given a fixed ppx (pixels per unit length) (x and z share the same
        #       ppx to assure aspect ratio is one), the corners of roi2 exactly 
        #       fall on integer based pixels. 
        roi1 = np.zeros((4, 3), dtype=float)
        roi2 = np.zeros((4, 3), dtype=float)
        w1 = np.linalg.norm(objPoints[0,:] - objPoints[2,:])
        h1 = np.linalg.norm(objPoints[0,:] - objPoints[1,:])
        roi1[0] = objPoints[0,:] 
        roi1[1] = roi1[0] + w1 * vecx
        roi1[2] = roi1[0] + w1 * vecx + h1 * vecz
        roi1[3] = roi1[0] + h1 * vecz

        # Estimate image location (must be integer) of P1 (intPosP1x/y)
        # and image size of rectified images (w_rectf, h_rectf)
        # The w_rectf and h_rectf do not change over time (istep)
        if istep == 0:
            intPosP1x = round(rexp * w1 * ppx)  
            intPosP1y = round((1. + rexp) * h1 * ppx)  
            w_rectf = round((1 + 2 * rexp) * w1 * ppx)  
            h_rectf = round((1 + 2 * rexp) * h1 * ppx)  
            print("# The initial P1 will be placed at pixel [%d,%d] ([y,x], zero based)"\
                  " of the initial rectified image" % (intPosP1y, intPosP1x))
            print("# Image width/height of metric rectification: (%d/%d) pixels." %
                  (w_rectf, h_rectf))

        # w2 should be very close to w1 * (1 + 2 * rexp) but is not exactly 
        # the same because w2 needs to be a multiple of (1./ppx) for integer based
        # number of pixels. So is h2. 
        w2 = (w_rectf - 1) * (1./ ppx)
        h2 = (h_rectf - 1) * (1./ ppx)
        roi2[3] = roi1[0] - intPosP1x / ppx * vecx + intPosP1y / ppx * vecz
        roi2[0] = roi2[3] - h2 * vecz
        roi2[1] = roi2[0] + w2 * vecx
        roi2[2] = roi2[1] + h2 * vecz
        
        # generate 3D points in a grid for remapping (gridPoints3D)
        # within 4 corners of roi2
        # Re-form formula (to matrix form): 0.047 sec. (h_rectf,w_rectf=2522,1671)
        # See perftest_rectf1.py for more performance test details.
        tic = time.time()
        mij = np.zeros((2, h_rectf * w_rectf), dtype=float)
        mij0, mij1 = np.meshgrid(range(w_rectf), range(h_rectf))
        mij[0] = mij0.flatten()
        mij[1] = mij1.flatten()
        mvec = np.array([vecx / ppx, -vecz / ppx]).transpose()
        mgx = np.matmul(mvec, mij) 
        grid3d = mgx.transpose() + roi2[3]
        toc = time.time()
        print("# It took %f sec. to calculate dense grid 3D coordinates." % (toc - tic))

        # generate and write rectified images
        rectfs = []
        for icam in range(2):
            # project 3D points to image coordinates.
            tic = time.time()
            testCase = 2 # 1: only single-thread, 2: only multithread, 3. both
            if testCase == 1:
                grid2d_icam, proj_jacob_icam =\
                      cv.projectPoints(grid3d, rvecs[icam], tvecs[icam], 
                                      cmats[icam], dvecs[icam])
            if testCase == 2:
                grid2d_icam, proj_jacob_icam =\
                    projectPoints_mp(grid3d, rvecs[icam], tvecs[icam], 
                                    cmats[icam], dvecs[icam])
            if testCase == 3:
                __grid2d_icam, __proj_jacob_icam =\
                      cv.projectPoints(grid3d, rvecs[icam], tvecs[icam], 
                                      cmats[icam], dvecs[icam])
                grid2d_icam, proj_jacob_icam =\
                    projectPoints_mp(grid3d, rvecs[icam], tvecs[icam], 
                                    cmats[icam], dvecs[icam])
                errmax1 = np.linalg.norm((__grid2d_icam - grid2d_icam).flatten())
                errmax2 = np.linalg.norm((__proj_jacob_icam - proj_jacob_icam).flatten())
                print("# Check projectPoint_mp: %f %f" % (errmax1, errmax2))
                if abs(errmax1) > 1e-6 or abs(errmax2) > 1e-6:
                    print("# Error: Multithreading leads to significant difference.")
            grid2d_icam = grid2d_icam.reshape(-1,2)
            toc = time.time()
            print("# It took %f sec. to project dense grid image (2D) coordinates." % (toc - tic))
        
            # Generate remapping coordinates
            map_x = grid2d_icam[:,0].reshape(h_rectf, w_rectf).astype(np.float32)
            map_y = grid2d_icam[:,1].reshape(h_rectf, w_rectf).astype(np.float32)
    
            # Generate rectification (--> rectfs or rectfs_tmplt)
            tic = time.time()
            rectf = cv.remap(imgs[icam], map_x, map_y, cv.INTER_LANCZOS4)
            toc = time.time()
            print("# It took %f sec. to remap rectified image (Cam:%d, step %d)."
                  % (toc - tic, icam + 1, istep + 1))
            if istep == 0:
                rectfs_tmplt.append(rectf)
            rectfs.append(rectf)
 
            # save rectification images to _wdir + 'rectf'
            if type(_wdir) == type(None):
                print('# Enter working directory (for output):')
                _wdir = input2()
            wdir_rectf = os.path.join(_wdir, 'rectf')
            if os.path.exists(wdir_rectf) == False:
                os.makedirs(wdir_rectf)
            tic = time.time()
            rectf_filename = os.path.join(wdir_rectf, 
                             'Step_%d_Cam_%d_Rectf.tiff' % (istep + 1, icam + 1))
            cv.imwrite(rectf_filename, rectf)
            toc = time.time()
            print("# It took %f sec. to write rectified image (Cam:%d, step %d)."
                  % (toc - tic, icam + 1, istep + 1))

        # run optical flow 
        #   cellSizes are estimated interval of neighboring optical flow points
        #    The actual interval could be slightly different because ROI 
        #    is fixed and number of optical flow points must be integers.
#        cellSizes = [15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
        if type(_cellSizes) == type(None):
            print("# Enter cell sizes you want to do displacement analysis.")
            print("#   Cell sizes are measured by pixels.")
            print("#   A small cell size induces refined mesh, more optical flow points, longer optical flow analysis time.")
            print("#   You can input multiple values so that this program does all of them but takes longer time.")
            print("#   E.g., 10 30 60")
            _cellSizes = input2()
        cellSizes = np.fromstring(_cellSizes, sep=' ', dtype=np.int32)
#        cellSizes = [15, 30, 60]
        # x0 y0 x1 y1 are the image coord. of rectified image of the roi
        x0 = intPosP1x * 1.0
        y0 = intPosP1y - h1 * ppx
        x1 = x0 + w1 * ppx
        y1 = intPosP1y * 1.0
        for iCellSize in cellSizes:
            for icam in range(2):
                # . calculate image points of templates
                nOpfPtsX = round((x1 - x0) / iCellSize)
                nOpfPtsY = round((y1 - y0) / iCellSize)
                prevPts = np.zeros((nOpfPtsY, nOpfPtsX, 2), dtype=np.float32)
                prevPts[:,:,0], prevPts[:,:,1] =\
                    np.meshgrid(np.linspace(x0, x1, nOpfPtsX),
                                np.linspace(y0, y1, nOpfPtsY))
                prevPts = prevPts.reshape(-1, 2)
                # prediction
                nextPts = prevPts.copy()
                # optical flow for current point positions (imgPts)
                tic = time.time()
                nextPts, status, err = cv.calcOpticalFlowPyrLK(
                    rectfs_tmplt[icam], 
                    rectfs[icam], 
                    prevPts,  
                    nextPts, 
                    winSize= (3 * iCellSize, 3 * iCellSize),
                    maxLevel=4,
                    criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 50, 0.001),
                    minEigThreshold=1e-4)
                toc = time.time()
                print("# It took %f sec. to run optical flow (Cam:%d, step %d)." % (toc - tic, icam + 1, istep + 1))
                # save optical flow result 
                nextPts = nextPts.reshape(nOpfPtsY, -1)
#                imgPts = nextPts.reshape(nOpfPtsY, nOpfPtsX, 2)
                wdir_fields = os.path.join(_wdir, 'fields')
                if os.path.exists(wdir_fields) == False:
                    os.makedirs(wdir_fields)
                tic = time.time()
                posi_file = os.path.join(wdir_fields, 'Step_%d_Cam_%d_imgPts_cellSize_%d.csv' % (istep + 1, icam + 1, iCellSize))
                np.savetxt(posi_file, nextPts, delimiter=' , ')
                toc = time.time()
                print("# It took %f sec. to write imgPts file (Cam:%d, step %d)."
                      % (toc - tic, icam + 1, istep + 1))
                # from point position (imgPts) to ux and uy (field_ux/y)
                field_ux = np.zeros((nOpfPtsY, nOpfPtsX), dtype=np.float32)
                field_uy = np.zeros((nOpfPtsY, nOpfPtsX), dtype=np.float32)
                field_uxy = (nextPts.flatten() - prevPts.flatten()).reshape((nOpfPtsY, nOpfPtsX, 2)) / ppx
                field_ux = field_uxy[:,:,0]
                field_uy = field_uxy[:,:,1]
                # save ux and uy files
                tic = time.time()
                field_ux_file = os.path.join(wdir_fields, 'Step_%d_Cam_%d_ux_cellSize_%d.csv' % (istep + 1, icam + 1, iCellSize))
                np.savetxt(field_ux_file, field_ux, delimiter=' , ')
                field_uy_file = os.path.join(wdir_fields, 'Step_%d_Cam_%d_uy_cellSize_%d.csv' % (istep + 1, icam + 1, iCellSize))
                np.savetxt(field_uy_file, field_uy, delimiter=' , ')
                toc = time.time()
                print("# It took %f sec. to write fields ux and uy file  (Cam:%d, step %d)."
                      % (toc - tic, icam + 1, istep + 1))
                # # from point position (imgPts) to ux and uy (field_ux/y)
                # field_ux = np.zeros((nOpfPtsY, nOpfPtsX), dtype=np.float32)
                # field_uy = np.zeros((nOpfPtsY, nOpfPtsX), dtype=np.float32)
                # field_uxy = (nextPts.flatten() - prevPts.flatten()).reshape((nOpfPtsY, nOpfPtsX, 2))
                # field_ux = field_uxy[:,:,0]
                # field_uy = field_uxy[:,:,1]
                # tic = time.time()
                # file = os.path.join(wdir_fields, 'field_ux_c%d_step%d_cellSize_%d_%dx%d.csv' % (icam + 1, istep + 1, iCellSize, nOpfPtsY, nOpfPtsX))
                # file = os.path.join(wdir_fields, 'Step_%d_Cam_%d_uy_cellSize_%d.csv' % (istep + 1, icam + 1, iCellSize))
                # np.savetxt(file, field_ux, delimiter=' , ')
                # file = os.path.join(wdir_fields, 'field_uy_c%d_step%d_cellSize_%d_%dx%d.csv' % (icam + 1, istep + 1, iCellSize, nOpfPtsY, nOpfPtsX))
                # np.savetxt(file, field_uy, delimiter=' , ')
                # toc = time.time()
                # print("# It took %f sec. to write fields ux and uy files  (Cam:%d, step %d)."
                #       % (toc - tic, icam + 1, istep + 1))
                # generate a virtually expanded displacement for later strain calculation
                tic = time.time()
                field_ux_exp = np.zeros((nOpfPtsY + 2, nOpfPtsX + 2), dtype=np.float32)
                field_ux_exp[1:-1,1:-1] = field_ux
                field_ux_exp[ 0,1:-1] = 2 * field_ux[ 0,:] - field_ux[ 1,:]
                field_ux_exp[-1,1:-1] = 2 * field_ux[-1,:] - field_ux[-2,:]
                field_ux_exp[1:-1, 0] = 2 * field_ux[:, 0] - field_ux[:, 1]
                field_ux_exp[1:-1,-1] = 2 * field_ux[:,-1] - field_ux[:,-2]
                field_ux_exp[ 0, 0] = 2 * field_ux[ 0, 0] - field_ux[ 1, 1]
                field_ux_exp[ 0,-1] = 2 * field_ux[ 0,-1] - field_ux[ 1,-2]
                field_ux_exp[-1, 0] = 2 * field_ux[-1, 0] - field_ux[-2, 1]
                field_ux_exp[-1,-1] = 2 * field_ux[-1,-1] - field_ux[-2,-2]
                # field_ux_exp[ 0,1:-1] = field_ux[ 0,:]
                # field_ux_exp[-1,1:-1] = field_ux[-1,:]
                # field_ux_exp[1:-1, 0] = field_ux[:, 0] 
                # field_ux_exp[1:-1,-1] = field_ux[:,-1] 
                # field_ux_exp[ 0, 0] = field_ux[ 0, 0]
                # field_ux_exp[ 0,-1] = field_ux[ 0,-1]
                # field_ux_exp[-1, 0] = field_ux[-1, 0]
                # field_ux_exp[-1,-1] = field_ux[-1,-1]
                field_uy_exp = np.zeros((nOpfPtsY + 2, nOpfPtsX + 2), dtype=np.float32)
                field_uy_exp[1:-1,1:-1] = field_uy
                field_uy_exp[ 0,1:-1] = 2 * field_uy[ 0,:] - field_uy[ 1,:]
                field_uy_exp[-1,1:-1] = 2 * field_uy[-1,:] - field_uy[-2,:]
                field_uy_exp[1:-1, 0] = 2 * field_uy[:, 0] - field_uy[:, 1]
                field_uy_exp[1:-1,-1] = 2 * field_uy[:,-1] - field_uy[:,-2]
                field_uy_exp[ 0, 0] = 2 * field_uy[ 0, 0] - field_uy[ 1, 1]
                field_uy_exp[ 0,-1] = 2 * field_uy[ 0,-1] - field_uy[ 1,-2]
                field_uy_exp[-1, 0] = 2 * field_uy[-1, 0] - field_uy[-2, 1]
                field_uy_exp[-1,-1] = 2 * field_uy[-1,-1] - field_uy[-2,-2]
                # field_uy_exp[ 0,1:-1] = field_uy[ 0,:]
                # field_uy_exp[-1,1:-1] = field_uy[-1,:]
                # field_uy_exp[1:-1, 0] = field_uy[:, 0] 
                # field_uy_exp[1:-1,-1] = field_uy[:,-1] 
                # field_uy_exp[ 0, 0] = field_uy[ 0, 0]
                # field_uy_exp[ 0,-1] = field_uy[ 0,-1]
                # field_uy_exp[-1, 0] = field_uy[-1, 0]
                # field_uy_exp[-1,-1] = field_uy[-1,-1]
                toc = time.time()
                print("# It took %f sec. to calculate expanded displacement (Cam:%d, step %d)."
                      % (toc - tic, icam + 1, istep + 1))
                # calculate strain fields
                # from ux and uy (field_ux/y) to strain (exx, eyy, gxy)
                # gxy is engineering shear strain: gamma_xy := d(ux)/dy + d(uy)/dx
                # (roughly roughly speaking gxy is not exy. gxy = exy + eyx = 2 * exy)
                usePythonForStrainFields = False
                # calculate strain fields by using native python form (without numba)
                dx = (x1 - x0) / nOpfPtsX / ppx
                dy = (y1 - y0) / nOpfPtsY / ppx
                if usePythonForStrainFields:
                    tic = time.time()
                    # calculate strain fields
                    # exx is d(ux)/dx, eyy is d(uy)/dy
                    # gxy is d(ux)/dy + d(uy)/dx, (not 0.5 d(ux)/dy + 0.5 d(uy)/dx)
                    # e1 and e2 are principal strains (2D) 
                    # gmx is maximum shear strain
                    # th1 and th2 are angles of e1 and e2 respectively (-180 to 180)
                    # thg is the angle of maximum shear strain (-180 to 180)
                    # angles are closewise from axis x, measured in degrees
                    field_exx = np.zeros((nOpfPtsY, nOpfPtsX), dtype=np.float32)
                    field_eyy = np.zeros((nOpfPtsY, nOpfPtsX), dtype=np.float32)
                    field_gxy = np.zeros((nOpfPtsY, nOpfPtsX), dtype=np.float32)
                    field_e1  = np.zeros((nOpfPtsY, nOpfPtsX), dtype=np.float32)
                    field_e2  = np.zeros((nOpfPtsY, nOpfPtsX), dtype=np.float32)
                    field_gmx = np.zeros((nOpfPtsY, nOpfPtsX), dtype=np.float32)
                    field_th1 = np.zeros((nOpfPtsY, nOpfPtsX), dtype=np.float32)
                    field_th2 = np.zeros((nOpfPtsY, nOpfPtsX), dtype=np.float32)
                    field_thg = np.zeros((nOpfPtsY, nOpfPtsX), dtype=np.float32)
                    for i in range(nOpfPtsY):
                        for j in range(nOpfPtsX):
                            ii = i + 1
                            jj = j + 1
                            exx = (field_ux_exp[ii,jj+1] - field_ux_exp[ii,jj-1]) / (2 * dx)
                            eyy = (field_uy_exp[ii-1,jj] - field_uy_exp[ii+1,jj]) / (2 * dy)
                            gxy = (field_ux_exp[ii-1,jj] - field_ux_exp[ii+1,jj]) / (2 * dy)\
                                + (field_uy_exp[ii,jj+1] - field_uy_exp[ii,jj-1]) / (2 * dx)
                            R = sqrt( ((exx - eyy) / 2.) ** 2 + (gxy / 2.) ** 2)
                            C = (exx + eyy) / 2.
                            e1 = C + R
                            e2 = C - R
                            gmx = 2 * R
                            th1 = 0.5 * atan2(-gxy, exx - eyy) * 180. / pi
                            th2 = 0.5 * atan2(gxy, -exx + eyy) * 180. / pi
                            thg = (45. - th1) % 360 - 180.
                            field_exx[i,j] = exx
                            field_eyy[i,j] = eyy
                            field_gxy[i,j] = gxy
                            field_e1[i,j]  = e1
                            field_e2[i,j]  = e2
                            field_gmx[i,j] = gmx
                            field_th1[i,j] = th1
                            field_th2[i,j] = th2
                            field_thg[i,j] = thg
                            toc = time.time()
                    print("# It took %f sec. to calculate fields exx, exy, and gxy (engineering shear strain gamma) files  (Cam:%d, step %d)."
                          % (toc - tic, icam + 1, istep + 1))
                # calculate crack by using numba
                useNumbaForStrainFields = True
                if useNumbaForStrainFields == True:
                    tic = time.time()
                    if usePythonForStrainFields == False:
                        field_exx, field_eyy, field_gxy, field_e1, field_e2,\
                            field_gmx, field_th1, field_th2, field_thg = \
                            calcStrainFields(field_ux, field_uy, dx, dy)
                    else:
                        _field_exx, _field_eyy, _field_gxy, _field_e1, _field_e2,\
                            _field_gmx, _field_th1, _field_th2, _field_thg = \
                            calcStrainFields(field_ux, field_uy, dx, dy)
                    toc = time.time()
                    print("# It took %f sec. to calculate fields exx, exy, and gxy by numba (engineering shear strain gamma) files  (Cam:%d, step %d)."
                          % (toc - tic, icam + 1, istep + 1))
                # check
                if usePythonForStrainFields and useNumbaForStrainFields:
                    if istep >= 1:
                        print("# Debug: strain check. istep, iCellSize, icam: %d,%d,%d" % (istep, iCellSize, icam))
                        _err_field_exx = np.linalg.norm((field_exx-_field_exx).flatten())
                        print("# Debug: norm of field_exx: %f" % np.linalg.norm(field_exx.flatten()))
                        print("# Error check between field_exx and _field_eyy: %f" % _err_field_exx)
                        _err_field_eyy = np.linalg.norm((field_eyy-_field_eyy).flatten())
                        print("# Debug: norm of field_eyy: %f" % np.linalg.norm(field_eyy.flatten()))
                        print("# Error check between field_eyy and _field_eyy: %f" % _err_field_eyy)
                        _err_field_gxy = np.linalg.norm((field_gxy-_field_gxy).flatten())
                        print("# Debug: norm of field_gxy: %f" % np.linalg.norm(field_gxy.flatten()))
                        print("# Error check between field_gxy and _field_gxy %f" % _err_field_gxy)
                        print("# ")
                    # copy numba result to field_crack
                    field_exx = _field_exx
                    field_eyy = _field_eyy
                    field_gxy = _field_gxy

                # write exx, eyy, exy, e1, e2, gmx, th1, th2, thg to files
                tic = time.time()
                field_exx_file = os.path.join(wdir_fields, 'Step_%d_Cam_%d_exx_cellSize_%d.csv' % (istep + 1, icam + 1, iCellSize))
                np.savetxt(field_exx_file, field_exx, delimiter=' , ')
                field_eyy_file = os.path.join(wdir_fields, 'Step_%d_Cam_%d_eyy_cellSize_%d.csv' % (istep + 1, icam + 1, iCellSize))
                np.savetxt(field_eyy_file, field_eyy, delimiter=' , ')
                field_gxy_file = os.path.join(wdir_fields, 'Step_%d_Cam_%d_gxy_cellSize_%d.csv' % (istep + 1, icam + 1, iCellSize))
                np.savetxt(field_gxy_file, field_gxy, delimiter=' , ')
                field_e1_file = os.path.join(wdir_fields, 'Step_%d_Cam_%d_e1_cellSize_%d.csv' % (istep + 1, icam + 1, iCellSize))
                np.savetxt(field_e1_file, field_e1, delimiter=' , ')
                field_e2_file = os.path.join(wdir_fields, 'Step_%d_Cam_%d_e2_cellSize_%d.csv' % (istep + 1, icam + 1, iCellSize))
                np.savetxt(field_e2_file, field_e2, delimiter=' , ')
                field_gmx_file = os.path.join(wdir_fields, 'Step_%d_Cam_%d_gmx_cellSize_%d.csv' % (istep + 1, icam + 1, iCellSize))
                np.savetxt(field_gmx_file, field_gmx, delimiter=' , ')
                field_th1_file = os.path.join(wdir_fields, 'Step_%d_Cam_%d_th1_cellSize_%d.csv' % (istep + 1, icam + 1, iCellSize))
                np.savetxt(field_th1_file, field_th1, delimiter=' , ')
                field_th2_file = os.path.join(wdir_fields, 'Step_%d_Cam_%d_th2_cellSize_%d.csv' % (istep + 1, icam + 1, iCellSize))
                np.savetxt(field_th2_file, field_th2, delimiter=' , ')
                field_thg_file = os.path.join(wdir_fields, 'Step_%d_Cam_%d_thg_cellSize_%d.csv' % (istep + 1, icam + 1, iCellSize))
                np.savetxt(field_thg_file, field_thg, delimiter=' , ')
                toc = time.time()
                print("# It took %f sec. to write fields exx, exy, gxy (engineering shear strain gamma), e1, e2 (principal strains), gmx (maximum shear strain), th1, th2, thg (angles of e1, e2, and gmx) files  (Cam:%d, step %d)."
                      % (toc - tic, icam + 1, istep + 1))
                # from ux and uy (field_ux/y) to crack opening
                usePythonForCrack = False
                # calculate crack by using native python form (without numba)
                if usePythonForCrack == True:
                    tic = time.time()
                    field_crack = np.zeros((nOpfPtsY, nOpfPtsX), dtype=np.float32)
                    for i in range(nOpfPtsY):
                        for j in range(nOpfPtsX):
                            maxCrack = -1e30
                            ii = i + 1
                            jj = j + 1
                            uU = np.array([field_ux_exp[ii-1, jj], field_uy_exp[ii-1, jj]], dtype=np.float32)
                            uD = np.array([field_ux_exp[ii+1, jj], field_uy_exp[ii+1, jj]], dtype=np.float32)
                            uL = np.array([field_ux_exp[ii, jj-1], field_uy_exp[ii, jj-1]], dtype=np.float32)
                            uR = np.array([field_ux_exp[ii, jj+1], field_uy_exp[ii, jj+1]], dtype=np.float32)
                            for th in [0, 45, 90, 135]:
                                costh = cos(th * np.pi / 180.)
                                sinth = sin(th * np.pi / 180.)
                                if th < 90:
                                    uA = (uU * abs(costh) + uL * abs(sinth)) / (abs(costh) + abs(sinth))
                                    uB = (uD * abs(costh) + uR * abs(sinth)) / (abs(costh) + abs(sinth))
                                else:
                                    uA = (uD * abs(costh) + uL * abs(sinth)) / (abs(costh) + abs(sinth))
                                    uB = (uU * abs(costh) + uR * abs(sinth)) / (abs(costh) + abs(sinth))
                                field_crackOpening =  -sinth * (uA[0] - uB[0]) - costh * (uA[1] - uB[1])
                                if field_crackOpening > maxCrack:
                                    maxCrack = field_crackOpening
                                    field_crack[i,j] = field_crackOpening
                    toc = time.time()
                    print("# It took %f sec. to analyze crack field  (Cam:%d, step %d)."
                          % (toc - tic, icam + 1, istep + 1))
                # calculate crack by using numba
                useNumbaForCrack = True
                if useNumbaForCrack == True:
                    tic = time.time()
                    if usePythonForCrack == False:
                        field_crack = calcCrackField(field_ux, field_uy)
                    else:
                        _field_crack = calcCrackField(field_ux, field_uy)
                    toc = time.time()
                    print("# It took %f sec. to analyze crack field  (Cam:%d, step %d)."
                          % (toc - tic, icam + 1, istep + 1))
                # check
                if usePythonForCrack and useNumbaForCrack:
                    if np.linalg.norm(field_crack.flatten()) > 0.1:
                        _err_field_crack = np.linalg.norm((field_crack-_field_crack).flatten())
                        print("# Debug: istep, iCellSize, icam: %d,%d,%d" % (istep, iCellSize, icam))
                        print("# Debug: norm of crack field: %f" % np.linalg.norm(field_crack.flatten()))
                        print("# Debug: norm of _crack field: %f" % np.linalg.norm(_field_crack.flatten()))
                        print("# Error check between field_crack and _field_crack: %f" % _err_field_crack)
                        print("# ")
                    # copy numba result to field_crack
                    field_crack = _field_crack
                # write crack result to file
                tic = time.time()
                field_crack_file = os.path.join(wdir_fields, 'Step_%d_Cam_%d_crack_cellSize_%d.csv' % (istep + 1, icam + 1, iCellSize))
                np.savetxt(field_crack_file, field_crack, delimiter=' , ')
                toc = time.time()
                print("# It took %f sec. to write fields crack (crack opening) file  (Cam:%d, step %d)."
                      % (toc - tic, icam + 1, istep + 1))
                
                # generate photos with fields
                tic = time.time()
                # # ux
                bg = rectf_filename
                ff = field_ux_file
                pf = posi_file
                cm = 'jet'
                rm = '0'
                co = ' -1 -1 -1 '
                vw = -1
                cl = ' -50 50 '
                sf = wdir_fields + '/Cam_%d_ux_cellSize_%d_cl30/Step_%d_Cam_%d_ux_cellSize_%d_cl50.JPG' % (icam+1, iCellSize, istep+1, icam+1, iCellSize)
#                viewField(bg,ff,pf,sf,cm,rm,co,cl,vw)
                drawField(bg,ff,pf,sf,cm,rm,co,cl,vw)
                cl = ' -10 10 '
                sf = wdir_fields + '/Cam_%d_ux_cellSize_%d_cl10/Step_%d_Cam_%d_ux_cellSize_%d_cl10.JPG' % (icam+1, iCellSize, istep+1, icam+1, iCellSize)
#                viewField(bg,ff,pf,sf,cm,rm,co,cl,vw)
                drawField(bg,ff,pf,sf,cm,rm,co,cl,vw)
                # uy
                bg = rectf_filename
                ff = field_uy_file
                pf = posi_file
                cm = 'jet'
                rm = '0'
                co = ' -1 -1 -1 '              
                vw = -1
                cl = ' -10 10 '
                sf = wdir_fields + '/Cam_%d_uy_cellSize_%d_cl10/Step_%d_Cam_%d_uy_cellSize_%d_cl10.JPG' % (icam+1, iCellSize, istep+1, icam+1, iCellSize)
#                viewField(bg,ff,pf,sf,cm,rm,co,cl,vw)
                drawField(bg,ff,pf,sf,cm,rm,co,cl,vw)
                # exx
                bg = rectf_filename
                ff = field_exx_file
                pf = posi_file
                cm = 'jet'
                rm = '0'
                co = ' -1 -1 -1 '
                vw = -1
                cl = ' -0.1 0.1 '
                sf = wdir_fields + '/Cam_%d_exx_cellSize_%d_cl100/Step_%d_Cam_%d_exx_cellSize_%d_cl100.JPG' % (icam+1, iCellSize, istep+1, icam+1, iCellSize)
#                viewField(bg,ff,pf,sf,cm,rm,co,cl,vw)
                drawField(bg,ff,pf,sf,cm,rm,co,cl,vw)
                cl = ' -0.01 0.01'
                sf = wdir_fields + '/Cam_%d_exx_cellSize_%d_cl010/Step_%d_Cam_%d_exx_cellSize_%d_cl010.JPG' % (icam+1, iCellSize, istep+1, icam+1, iCellSize)
#                viewField(bg,ff,pf,sf,cm,rm,co,cl,vw)
                drawField(bg,ff,pf,sf,cm,rm,co,cl,vw)
                # cl = ' -0.001 0.001 '
                # sf = wdir_fields + '/Cam_%d_exx_cellSize_%d_cl001/Step_%d_Cam_%d_exx_cellSize_%d_cl001.JPG' % (icam+1, iCellSize, istep+1, icam+1, iCellSize)
                # viewField(bg,ff,pf,sf,cm,rm,co,cl,vw)
                # eyy
                bg = rectf_filename
                ff = field_eyy_file
                pf = posi_file
                cm = 'jet'
                rm = '0'
                co = ' -1 -1 -1 '              
                vw = -1
                cl = ' -0.1 0.1 '
                sf = wdir_fields + '/Cam_%d_eyy_cellSize_%d_cl100/Step_%d_Cam_%d_eyy_cellSize_%d_cl100.JPG' % (icam+1, iCellSize, istep+1, icam+1, iCellSize)
#                viewField(bg,ff,pf,sf,cm,rm,co,cl,vw)
                drawField(bg,ff,pf,sf,cm,rm,co,cl,vw)
                cl = ' -0.01 0.01'
                sf = wdir_fields + '/Cam_%d_eyy_cellSize_%d_cl010/Step_%d_Cam_%d_eyy_cellSize_%d_cl010.JPG' % (icam+1, iCellSize, istep+1, icam+1, iCellSize)
#                viewField(bg,ff,pf,sf,cm,rm,co,cl,vw)
                drawField(bg,ff,pf,sf,cm,rm,co,cl,vw)
                # cl = ' -0.001 0.001 '
                # sf = wdir_fields + '/Cam_%d_eyy_cellSize_%d_cl001/Step_%d_Cam_%d_eyy_cellSize_%d_cl001.JPG' % (icam+1, iCellSize, istep+1, icam+1, iCellSize)
                # viewField(bg,ff,pf,sf,cm,rm,co,cl,vw)
                # gxy
                bg = rectf_filename
                ff = field_gxy_file
                pf = posi_file
                cm = 'jet'
                rm = '0'
                co = ' -1 -1 -1 '              
                vw = -1
                cl = ' -0.1 0.1 '
                sf = wdir_fields + '/Cam_%d_gxy_cellSize_%d_cl100/Step_%d_Cam_%d_gxy_cellSize_%d_cl100.JPG' % (icam+1, iCellSize, istep+1, icam+1, iCellSize)
#                viewField(bg,ff,pf,sf,cm,rm,co,cl,vw)
                drawField(bg,ff,pf,sf,cm,rm,co,cl,vw)
                cl = ' -0.01 0.01'
                sf = wdir_fields + '/Cam_%d_gxy_cellSize_%d_cl010/Step_%d_Cam_%d_gxy_cellSize_%d_cl010.JPG' % (icam+1, iCellSize, istep+1, icam+1, iCellSize)
#                viewField(bg,ff,pf,sf,cm,rm,co,cl,vw)
                drawField(bg,ff,pf,sf,cm,rm,co,cl,vw)
                # cl = ' -0.001 0.001 '
                # sf = wdir_fields + '/Cam_%d_gxy_cellSize_%d_cl001/Step_%d_Cam_%d_gxy_cellSize_%d_cl001.JPG' % (icam+1, iCellSize, istep+1, icam+1, iCellSize)
                # viewField(bg,ff,pf,sf,cm,rm,co,cl,vw)
                # e1
                bg = rectf_filename
                ff = field_e1_file
                pf = posi_file
                cm = 'jet'
                rm = '0'
                co = ' -1 -1 -1 '              
                vw = -1
                cl = ' -0.1 0.1 '
                sf = wdir_fields + '/Cam_%d_e1_cellSize_%d_cl100/Step_%d_Cam_%d_e1_cellSize_%d_cl100.JPG' % (icam+1, iCellSize, istep+1, icam+1, iCellSize)
#                viewField(bg,ff,pf,sf,cm,rm,co,cl,vw)
                drawField(bg,ff,pf,sf,cm,rm,co,cl,vw)
                cl = ' -0.01 0.01'
                sf = wdir_fields + '/Cam_%d_e1_cellSize_%d_cl010/Step_%d_Cam_%d_e1_cellSize_%d_cl010.JPG' % (icam+1, iCellSize, istep+1, icam+1, iCellSize)
#                viewField(bg,ff,pf,sf,cm,rm,co,cl,vw)
                drawField(bg,ff,pf,sf,cm,rm,co,cl,vw)
                # e2
                bg = rectf_filename
                ff = field_e2_file
                pf = posi_file
                cm = 'jet'
                rm = '0'
                co = ' -1 -1 -1 '              
                vw = -1
                cl = ' -0.1 0.1 '
                sf = wdir_fields + '/Cam_%d_e2_cellSize_%d_cl100/Step_%d_Cam_%d_e2_cellSize_%d_cl100.JPG' % (icam+1, iCellSize, istep+1, icam+1, iCellSize)
#                viewField(bg,ff,pf,sf,cm,rm,co,cl,vw)
                drawField(bg,ff,pf,sf,cm,rm,co,cl,vw)
                cl = ' -0.01 0.01'
                sf = wdir_fields + '/Cam_%d_e2_cellSize_%d_cl010/Step_%d_Cam_%d_e2_cellSize_%d_cl010.JPG' % (icam+1, iCellSize, istep+1, icam+1, iCellSize)
#                viewField(bg,ff,pf,sf,cm,rm,co,cl,vw)
                drawField(bg,ff,pf,sf,cm,rm,co,cl,vw)
                # gmx (maximum shear strain)
                bg = rectf_filename
                ff = field_gmx_file
                pf = posi_file
                cm = 'jet'
                rm = '0'
                co = ' -1 -1 -1 '              
                vw = -1
                cl = ' -0.1 0.1 '
                sf = wdir_fields + '/Cam_%d_gmx_cellSize_%d_cl100/Step_%d_Cam_%d_gmx_cellSize_%d_cl100.JPG' % (icam+1, iCellSize, istep+1, icam+1, iCellSize)
#                viewField(bg,ff,pf,sf,cm,rm,co,cl,vw)
                drawField(bg,ff,pf,sf,cm,rm,co,cl,vw)
                cl = ' -0.01 0.01'
                sf = wdir_fields + '/Cam_%d_gmx_cellSize_%d_cl010/Step_%d_Cam_%d_gmx_cellSize_%d_cl010.JPG' % (icam+1, iCellSize, istep+1, icam+1, iCellSize)
#                viewField(bg,ff,pf,sf,cm,rm,co,cl,vw)
                drawField(bg,ff,pf,sf,cm,rm,co,cl,vw)
                # th1 (angle of e1)
                bg = rectf_filename
                ff = field_th1_file
                pf = posi_file
                cm = 'jet'
                rm = '0'
                co = ' -1 -1 -1 '              
                vw = -1
                cl = ' -180 180 '
                sf = wdir_fields + '/Cam_%d_th1_cellSize_%d_cl180/Step_%d_Cam_%d_th1_cellSize_%d_cl180.JPG' % (icam+1, iCellSize, istep+1, icam+1, iCellSize)
#                viewField(bg,ff,pf,sf,cm,rm,co,cl,vw)
                drawField(bg,ff,pf,sf,cm,rm,co,cl,vw)
                # th2 (angle of e1)
                bg = rectf_filename
                ff = field_th2_file
                pf = posi_file
                cm = 'jet'
                rm = '0'
                co = ' -1 -1 -1 '              
                vw = -1
                cl = ' -180 180 '
                sf = wdir_fields + '/Cam_%d_th2_cellSize_%d_cl180/Step_%d_Cam_%d_th2_cellSize_%d_cl180.JPG' % (icam+1, iCellSize, istep+1, icam+1, iCellSize)
#                viewField(bg,ff,pf,sf,cm,rm,co,cl,vw)
                drawField(bg,ff,pf,sf,cm,rm,co,cl,vw)
                # thg (angle of e1)
                bg = rectf_filename
                ff = field_thg_file
                pf = posi_file
                cm = 'jet'
                rm = '0'
                co = ' -1 -1 -1 '              
                vw = -1
                cl = ' -180 180 '
                sf = wdir_fields + '/Cam_%d_thg_cellSize_%d_cl180/Step_%d_Cam_%d_thg_cellSize_%d_cl180.JPG' % (icam+1, iCellSize, istep+1, icam+1, iCellSize)
#                viewField(bg,ff,pf,sf,cm,rm,co,cl,vw)
                drawField(bg,ff,pf,sf,cm,rm,co,cl,vw)

                # crack
                bg = rectf_filename
                ff = field_crack_file
                pf = posi_file
                cm = 'bone'
                rm = '1'
                co = ' -1 -1 -1 '
                vw = -1
                # cl = ' 0 5 '
                # sf = wdir_fields + '/Cam_%d_crack_cellSize_%d_cl50/Step_%d_Cam_%d_crack_cellSize_%d_cl50.JPG' % (icam+1, iCellSize, istep+1, icam+1, iCellSize)
                # viewField(bg,ff,pf,sf,cm,rm,co,cl,vw)
                cl = ' 0 1 '
                sf = wdir_fields + '/Cam_%d_crack_cellSize_%d_cl10/Step_%d_Cam_%d_crack_cellSize_%d_cl10.JPG' % (icam+1, iCellSize, istep+1, icam+1, iCellSize)
#                viewField(bg,ff,pf,sf,cm,rm,co,cl,vw)
                drawField(bg,ff,pf,sf,cm,rm,co,cl,vw)
                # cl = ' 0 0.2 '
                # sf = wdir_fields + '/Cam_%d_crack_cellSize_%d_cl02/Step_%d_Cam_%d_crack_cellSize_%d_cl02.JPG' % (icam+1, iCellSize, istep+1, icam+1, iCellSize)
                # viewField(bg,ff,pf,sf,cm,rm,co,cl,vw)

                # bgImgFile = r'D:\yuansen\ImPro\improMeasure\examples\2022rcwall_cracktest\rectf\Step_2_Cam_1_Rectf.tiff'
                # fieldFile = r'D:\yuansen\ImPro\improMeasure\examples\2022rcwall_cracktest\fields\Step_2_Cam_1_crack_cellSize_15.csv'
                # posiFile =  r'D:\yuansen\ImPro\improMeasure\examples\2022rcwall_cracktest\fields\Step_2_Cam_1_imgPts_cellSize_15.csv'
                # saveImgFile = r'D:\yuansen\ImPro\improMeasure\examples\2022rcwall_cracktest\fields\Step_2_Cam_1_crack_cellSize_15.JPG'
                # colormap = 'bone'
                # reverseMap = '1'
                # cOut = ' 255  255  255'
                # clim = '0 1'
                # viewField(_bgImgFile=bgImgFile,
                #           _fieldFile=fieldFile,
                #           _posiFile=posiFile,
                #           _saveImgFile=saveImgFile,
                #           _colormap=colormap,
                #           _reverseMap=reverseMap,
                #           _cOut=cOut,
                #           _clim=clim)
                toc = time.time()
                print("# It took %f sec. to draw fields on new rectf images and save them (Cam:%d, step %d)."
                      % (toc - tic, icam + 1, istep + 1))
                if istep > 1:
                    pass


def test5():
    icf_wallMonitor_v2(
        _files1=r'D:\ExpDataSamples\20220200_NcreeNorth_RcWall\Analysis_Specimen_1_Cams_1_2\Left\*.JPG',
        _files2=r'D:\ExpDataSamples\20220200_NcreeNorth_RcWall\Analysis_Specimen_1_Cams_1_2\Right\*.JPG',
        _cam1=r'D:\ExpDataSamples\20220200_NcreeNorth_RcWall\Calibration_all\Specimen01_Canon_L_20220310_102517_camera_parameters.csv',
        _cam2=r'D:\ExpDataSamples\20220200_NcreeNorth_RcWall\Calibration_all\Specimen01_Canon_R_20220310_102517_camera_parameters.csv',
        _ctrlp1file=r'D:\ExpDataSamples\20220200_NcreeNorth_RcWall\Analysis_improMeasure_Spc_1_Cams1_2\Canon_L_ctrlp.txt',
        _ctrlp2file=r'D:\ExpDataSamples\20220200_NcreeNorth_RcWall\Analysis_improMeasure_Spc_1_Cams1_2\Canon_R_ctrlp.txt',
        _maxMove='50 30',
        _ppx='2.0', 
        _xzzx='xz',
        _rexp='0.1',
        _cellSizes='15 30 60',
        _wdir=r'D:\ExpDataSamples\20220200_NcreeNorth_RcWall\Analysis_improMeasure_Spc_1_Cams1_2',
        )

    icf_wallMonitor_v2(
        _files1=r'D:\ExpDataSamples\20220200_NcreeNorth_RcWall\Analysis_Specimen_2_Cams_1_2\Left\*.JPG',
        _files2=r'D:\ExpDataSamples\20220200_NcreeNorth_RcWall\Analysis_Specimen_2_Cams_1_2\Right\*.JPG',
        _cam1=r'D:\ExpDataSamples\20220200_NcreeNorth_RcWall\Calibration_all\Specimen02_Canon_L_20220314_095830_camera_parameters.csv',
        _cam2=r'D:\ExpDataSamples\20220200_NcreeNorth_RcWall\Calibration_all\Specimen02_Canon_R_20220314_095830_camera_parameters.csv',
        _ctrlp1file=r'D:\ExpDataSamples\20220200_NcreeNorth_RcWall\Analysis_improMeasure_Spc_2_Cams1_2\Canon_L_ctrlp.txt',
        _ctrlp2file=r'D:\ExpDataSamples\20220200_NcreeNorth_RcWall\Analysis_improMeasure_Spc_2_Cams1_2\Canon_R_ctrlp.txt',
        _maxMove='50 30',
        _ppx='2.0', 
        _xzzx='xz',
        _rexp='0.1',
        _cellSizes='15 30 60',
        _wdir=r'D:\ExpDataSamples\20220200_NcreeNorth_RcWall\Analysis_improMeasure_Spc_2_Cams1_2'
        )

    icf_wallMonitor_v2(
        _files1=r'D:\ExpDataSamples\20220200_NcreeNorth_RcWall\Analysis_Specimen_3_Cams_1_2\Left\*.JPG',
        _files2=r'D:\ExpDataSamples\20220200_NcreeNorth_RcWall\Analysis_Specimen_3_Cams_1_2\Right\*.JPG',
        _cam1=r'D:\ExpDataSamples\20220200_NcreeNorth_RcWall\Calibration_all\Specimen03_Canon_L_20220321_101202_camera_parameters.csv',
        _cam2=r'D:\ExpDataSamples\20220200_NcreeNorth_RcWall\Calibration_all\Specimen03_Canon_R_20220321_101202_camera_parameters.csv',
        _ctrlp1file=r'D:\ExpDataSamples\20220200_NcreeNorth_RcWall\Analysis_improMeasure_Spc_3_Cams1_2\Canon_L_ctrlp.txt',
        _ctrlp2file=r'D:\ExpDataSamples\20220200_NcreeNorth_RcWall\Analysis_improMeasure_Spc_3_Cams1_2\Canon_R_ctrlp.txt',
        _maxMove='50 30',
        _ppx='2.0', 
        _xzzx='xz',
        _rexp='0.1',
        _cellSizes='15 30 60',
        _wdir=r'D:\ExpDataSamples\20220200_NcreeNorth_RcWall\Analysis_improMeasure_Spc_3_Cams1_2'
        )

    icf_wallMonitor_v2(
        _files1=r'D:\ExpDataSamples\20220200_NcreeNorth_RcWall\Analysis_Specimen_4_Cams_1_2\Left\*.JPG',
        _files2=r'D:\ExpDataSamples\20220200_NcreeNorth_RcWall\Analysis_Specimen_4_Cams_1_2\Right\*.JPG',
        _cam1=r'D:\ExpDataSamples\20220200_NcreeNorth_RcWall\Calibration_all\Specimen04_Canon_L_20220324_102932_camera_parameters.csv',
        _cam2=r'D:\ExpDataSamples\20220200_NcreeNorth_RcWall\Calibration_all\Specimen04_Canon_R_20220324_102932_camera_parameters.csv',
        _ctrlp1file=r'D:\ExpDataSamples\20220200_NcreeNorth_RcWall\Analysis_improMeasure_Spc_4_Cams1_2\Canon_L_ctrlp.txt',
        _ctrlp2file=r'D:\ExpDataSamples\20220200_NcreeNorth_RcWall\Analysis_improMeasure_Spc_4_Cams1_2\Canon_R_ctrlp.txt',
        _maxMove='50 30',
        _ppx='2.0', 
        _xzzx='xz',
        _rexp='0.1',
        _cellSizes='15 30 60',
        _wdir=r'D:\ExpDataSamples\20220200_NcreeNorth_RcWall\Analysis_improMeasure_Spc_4_Cams1_2'
        )

def test6():
    icf_wallMonitor_v2(
        _files1=r'D:\ExpDataSamples\20220200_NcreeNorth_RcWall\Analysis_improMeasure_Spc_1_Cams1_2_chk97\Left\*.JPG',
        _files2=r'D:\ExpDataSamples\20220200_NcreeNorth_RcWall\Analysis_improMeasure_Spc_1_Cams1_2_chk97\Right\*.JPG',
        _cam1=r'D:\ExpDataSamples\20220200_NcreeNorth_RcWall\Calibration_all\Specimen01_Canon_L_20220310_102517_camera_parameters.csv',
        _cam2=r'D:\ExpDataSamples\20220200_NcreeNorth_RcWall\Calibration_all\Specimen01_Canon_R_20220310_102517_camera_parameters.csv',
        _ctrlp1file=r'D:\ExpDataSamples\20220200_NcreeNorth_RcWall\Analysis_improMeasure_Spc_1_Cams1_2\Canon_L_ctrlp.txt',
        _ctrlp2file=r'D:\ExpDataSamples\20220200_NcreeNorth_RcWall\Analysis_improMeasure_Spc_1_Cams1_2\Canon_R_ctrlp.txt',
        _maxMove='50 30',
        _ppx='2.0', 
        _xzzx='xz',
        _rexp='0.1',
        _cellSizes='15 30 60',
        _wdir=r'D:\ExpDataSamples\20220200_NcreeNorth_RcWall\Analysis_improMeasure_Spc_1_Cams1_2_chk97',
        )


if __name__ == '__main__':
    tic = time.time()
    test5()        
    toc = time.time()
    print("### Total elapsed time: %f sec." % (toc - tic))
    