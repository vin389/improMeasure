#import numpy as np
import cv2 as cv
import numpy as np
import time


def demoOpticalFlow():
    #
    # Define image sources
    fileImg1 = r'D:\yuansen\ImPro\improMeasure\examples\2022rcwall\leftPause\DSC02766.JPG'
    #fileImg2 = r'D:\yuansen\ImPro\improMeasure\examples\2022rcwall\leftPause\DSC02766.JPG'
    fileImg2 = r'D:\yuansen\ImPro\improMeasure\examples\2022rcwall\leftPause\DSC03183.JPG'
#    y0, x0 = 2163, 3103
    yc, xc = 2160 + 75, 3140 + 75
    ht, wt = 40, 40
    sy, sx = 55, 55
    y0, x0 = yc - ht // 2, xc - wt // 2
    img1Full = cv.imread(fileImg1, cv.IMREAD_GRAYSCALE)
    img2Full = cv.imread(fileImg2, cv.IMREAD_GRAYSCALE)
#    img1 = img1Full[y0:y0+ht, x0:x0+wt].copy()
    tm = img1Full[y0:y0+ht, x0:x0+wt].copy()
    img1 = img1Full[y0-sy:y0+ht+sy, x0-sx:x0+wt+sx].copy() # as optical flow requires img1 (imgPrev) and img2 (imgNext) to be the same size
    img2 = img2Full[y0-sy:y0+ht+sy, x0-sx:x0+wt+sx].copy()
    ptsPrev = ((wt + 2. * sx - 1.) / 2., (ht + 2. * sy - 1.) / 2.)
    ptsPrev = np.array(ptsPrev, dtype=np.float32).reshape([1,1,2])
    ptsNext = ptsPrev.copy()
    # template match
    # tic = time.time()
    # corrTm = cv.matchTemplate(img2, tm, cv.TM_CCORR_NORMED)
    # toc = time.time()
    # print("Template match (%dx%d in %dx%d): %9.5f sec." %
    #       (img1.shape[1], img1.shape[0], img2.shape[1], img2.shape[0], toc - tic))
    # corrTmImg = cv.normalize(corrTm, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
    # cv.imshow("TM", corrTmImg); 
    # cv.waitKey(0); cv.destroyAllWindows()
    tic = time.time()
    for i in range(100):
        res = cv.calcOpticalFlowPyrLK(img1, img2, ptsPrev, ptsNext, winSize=(ht, wt), maxLevel=3, minEigThreshold=0.001)
    toc = time.time()
    print("Optical flow (wsize:%dx%d): %9.4f sec." %
          (wt, ht, (toc - tic) / 100.))


    # draw optical flow
    img3 = drawOpticalFlow(img1, img2, ptsPrev, ptsNext, winSize=(wt, ht)) 
    cv.imshow("Matches", img3); cv.waitKey(0); cv.destroyAllWindows()


def drawOpticalFlow(imgPrev, imgNext, ptsPrev, ptsNext, winSize):
    imgShow = np.zeros((imgPrev.shape[0], 2*imgPrev.shape[1]), dtype=imgPrev.dtype)
    # background
    imgShow[:,0:imgPrev.shape[1]] = imgPrev
    imgShow[:,imgPrev.shape[1]:] = imgNext
    # ptsPrev
    ptsPrev = ptsPrev.reshape(-1,1,2)
    npoint = ptsPrev.shape[0]
    for i in range(npoint):
        pt1 = np.array((ptsPrev[i,0,0] - winSize[0] / 2., ptsPrev[i,0,1] - winSize[1] / 2.), dtype=int)
        pt2 = np.array((ptsPrev[i,0,0] + winSize[0] / 2., ptsPrev[i,0,1] + winSize[1] / 2.), dtype=int)
        cv.rectangle(imgShow, pt1, pt2, color=0, thickness=2, shift=0)
    # ptsNext
    ptsNext = ptsNext.reshape(-1,1,2)
    npoint = ptsNext.shape[0]
    for i in range(npoint):
        pt1 = np.array((ptsNext[i,0,0] - winSize[0] / 2. + imgPrev.shape[1] + .5, ptsNext[i,0,1] - winSize[1] / 2. + .5), dtype=int)
        pt2 = np.array((ptsNext[i,0,0] + winSize[0] / 2. + imgPrev.shape[1] + .5, ptsNext[i,0,1] + winSize[1] / 2. + .5), dtype=int)
        cv.rectangle(imgShow, pt1, pt2, color=0, thickness=2, shift=0)
#    cv.imshow("Matches", imgShow); cv.waitKey(0); cv.destroyAllWindows()
    return imgShow

#demoOpticalFlow()

