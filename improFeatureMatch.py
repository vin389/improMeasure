#import numpy as np
import cv2 as cv
import time
import numpy as np


def demoFeatureMatch():
    # This function is modified from an OpenCV tutorial webpage: 
    #   https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
    #
    # Define image sources
    fileImg1 = r'D:\yuansen\ImPro\improMeasure\examples\2022rcwall\leftPause\DSC02766.JPG'
    #fileImg2 = r'D:\yuansen\ImPro\improMeasure\examples\2022rcwall\leftPause\DSC02766.JPG'
    #fileImg2 = r'D:\yuansen\ImPro\improMeasure\examples\2022rcwall\leftPause\DSC02815.JPG'
    #fileImg2 = r'D:\yuansen\ImPro\improMeasure\examples\2022rcwall\leftPause\DSC02925.JPG'
    fileImg2 = r'D:\yuansen\ImPro\improMeasure\examples\2022rcwall\leftPause\DSC03183.JPG'
    y0, x0 = 2160, 3140
    ht, wt = 150, 150
    sy, sx = 0, 0
    img1Full = cv.imread(fileImg1, cv.IMREAD_GRAYSCALE)
    img2Full = cv.imread(fileImg2, cv.IMREAD_GRAYSCALE)
    img1 = img1Full[y0:y0+ht, x0:x0+wt].copy()
    img2 = img2Full[y0-sy:y0+ht+sy, x0-sx:x0+wt+sx].copy()
    # template match
    tic = time.time()
    corrTm = cv.matchTemplate(img2, img1, cv.TM_CCORR_NORMED)
    toc = time.time()
    print("Template match (%dx%d in %dx%d): %9.5f sec." %
          (img1.shape[1], img1.shape[0], img2.shape[1], img2.shape[0], toc - tic))
    corrTmImg = cv.normalize(corrTm, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
    cv.imshow("TM", corrTmImg); 
    # cv.waitKey(0); cv.destroyAllWindows()
    # detector
    orb = cv.ORB_create()
    # keypoints and descriptor
    tic = time.time()
    kp1, des1 = orb.detectAndCompute(img1,None)
    toc = time.time()
    print("Detect and compute of img1 (%dx%d): %9.4f sec. (%d points)" %
          (img1.shape[1], img1.shape[0], toc - tic, len(kp1) ))
    tic = time.time()
    kp2, des2 = orb.detectAndCompute(img2,None)
    toc = time.time()
    print("Detect and compute of img2 (%dx%d): %9.4f sec. (%d points)" %
          (img2.shape[1], img2.shape[0], toc - tic, len(kp2) ))
    # create matcher
    tic = time.time()
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    toc = time.time()
    print("Create matcher: %9.4f sec." % (toc - tic))
    # match
    tic = time.time()
    matches = bf.match(des1,des2)
    toc = time.time()
    print("Match: %9.4f sec. (%d matches)" % (toc - tic, len(matches)))
    # sort
    tic = time.time()
    matches = sorted(matches, key = lambda x:x.distance)
    toc = time.time()
    print("Sort: %9.4f sec." % (toc - tic))
    # draw matches
    nPointsDraw = 30
    img3 = cv.drawMatches(img1,kp1,img2,kp2,matches[:nPointsDraw],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv.imshow("Matches", img3); cv.waitKey(0); cv.destroyAllWindows()
    # find homography
    src_pts = np.zeros((nPointsDraw, 2), dtype=np.float32)
    dst_pts = np.zeros((nPointsDraw, 2), dtype=np.float32)
    for i in range(nPointsDraw):
        src_pts[i,:] = kp1[matches[i].queryIdx].pt
        dst_pts[i,:] = kp2[matches[i].trainIdx].pt
    tic = time.time()
    warp, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    print("Warp matrix: \n", warp)
    toc = time.time()
    print("Find homography: %9.4f sec." % (toc - tic))
    
    # print first 5 matches
    for i in range(5):
        print("Match %d:" % (i + 1))
        p1 = matches[i].queryIdx
        p2 = matches[i].trainIdx
        print("  P1[%3d](%4d,%4d): " % (p1, kp1[p1].pt[0], kp1[p1].pt[1]), end='')
        for j in range(len(des1[p1])):
            print("%02x " % des1[p1][j], end='')
        print("")
        print("  P2[%3d](%4d,%4d): " % (p2, kp2[p2].pt[0], kp2[p2].pt[1]), end='')
        for j in range(len(des2[p2])):
            print("%02x " % des2[p2][j], end='')
        print("")
        
#        print("  P1[%d](%d,%d): %02x %02x %02x %02x %02x ... " % (p1, kp1[p1].pt[0], kp1[p1].pt[1],  
#              des1[p1][0], des1[p1][1], des1[p1][2], des1[p1][3], des1[p1][4]))
#        print("  P2[%d](%d,%d): %02x %02x %02x %02x %02x ... " % (p2, kp2[p2].pt[0], kp2[p2].pt[1],  
#              des2[p2][0], des2[p2][1], des2[p2][2], des2[p2][3], des2[p2][4]))

# demoFeatureMatch()
