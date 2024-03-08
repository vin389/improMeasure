#import numpy as np
import cv2 as cv
import numpy as np
import time


def demoTemplateMatch():
    #
    # Define image sources
    fileImg1 = r'D:\yuansen\ImPro\improMeasure\examples\2022rcwall\leftPause\DSC02766.JPG'
    #fileImg2 = r'D:\yuansen\ImPro\improMeasure\examples\2022rcwall\leftPause\DSC02766.JPG'
    #fileImg2 = r'D:\yuansen\ImPro\improMeasure\examples\2022rcwall\leftPause\DSC02883.JPG'
    #fileImg2 = r'D:\yuansen\ImPro\improMeasure\examples\2022rcwall\leftPause\DSC02945.JPG'
    #fileImg2 = r'D:\yuansen\ImPro\improMeasure\examples\2022rcwall\leftPause\DSC03063.JPG'
    #fileImg2 = r'D:\yuansen\ImPro\improMeasure\examples\2022rcwall\leftPause\DSC03120.JPG'
    fileImg2 = r'D:\yuansen\ImPro\improMeasure\examples\2022rcwall\leftPause\DSC03183.JPG'
    #fileImg2 = r'D:\yuansen\ImPro\improMeasure\examples\2022rcwall\leftPause\DSC02899R.JPG'
#    fileImg2 = r'D:\yuansen\ImPro\improMeasure\examples\2022rcwall\leftPause\DSC03037R.JPG'
#    y0, x0 = 2163, 3103
    yc, xc = 2160 + 75, 3140 + 75
    ht, wt = 40, 40
    sy, sx = 55, 55
    y0, x0 = yc - ht // 2, xc - wt // 2
    img1Full = cv.imread(fileImg1, cv.IMREAD_GRAYSCALE)
    img2Full = cv.imread(fileImg2, cv.IMREAD_GRAYSCALE)
    tm = img1Full[y0:y0+ht, x0:x0+wt].copy()
    img1 = img1Full[y0-sy:y0+ht+sy, x0-sx:x0+wt+sx].copy()
    img2 = img2Full[y0-sy:y0+ht+sy, x0-sx:x0+wt+sx].copy()
    # template match
    tic = time.time()
#    match_method = cv.TM_CCORR_NORMED
    match_method = cv.TM_CCOEFF_NORMED
    corrTm = cv.matchTemplate(img2, tm, match_method)
    toc = time.time()
    print("Template match (%dx%d in %dx%d): %9.5f sec." %
          (img1.shape[1], img1.shape[0], img2.shape[1], img2.shape[0], toc - tic))
    corrTmImg = cv.normalize(corrTm, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
    _minVal, _maxVal, minLoc, maxLoc = cv.minMaxLoc(corrTm, None)
    if (match_method == cv.TM_SQDIFF or match_method == cv.TM_SQDIFF_NORMED):
        matchLoc = minLoc
    else:
        matchLoc = maxLoc
    # find the best location based on 2nd order curve
    #   about x
    if matchLoc[0] - 2 >= 0 and matchLoc[0] + 2 < corrTm.shape[1]:
#        n = 5
        n = 3 # so far we only use n = 3 (taking 3 points, rather than 5 points)
        xmat = np.zeros((n,3), dtype=float)
        ymat = np.zeros((n,1), dtype=float)
        i = matchLoc[1]
        for j in range(n):
            xmat[j, 0] = (j - (n - 1) // 2) ** 2
            xmat[j, 1] = (j - (n - 1) // 2)
            xmat[j, 2] = 1.
            ymat[j, 0] = corrTm[i, (j - (n - 1) // 2) + matchLoc[0]]
        amat = np.matmul(np.matmul(np.linalg.inv(np.matmul(xmat.transpose(), xmat)), xmat.transpose()), ymat)
        xbest = matchLoc[0] + amat[1,0] / (-2. * amat[0,0])
    elif matchLoc[0] - 1 >= 0 and matchLoc[0] + 1 < corrTm.shape[1]:
        n = 3
        xmat = np.zeros((n,3), dtype=float)
        ymat = np.zeros((n,1), dtype=float)
        i = matchLoc[1]
        for j in range(n):
            xmat[j, 0] = (j - (n - 1) // 2) ** 2
            xmat[j, 1] = (j - (n - 1) // 2)
            xmat[j, 2] = 1.
            ymat[j, 0] = corrTm[i, (j - (n - 1) // 2) + matchLoc[0]]
        amat = np.matmul(np.matmul(np.linalg.inv(np.matmul(xmat.transpose(), xmat)), xmat.transpose()), ymat)
        xbest = matchLoc[0] + amat[1,0] / (-2. * amat[0,0])
    else:
        xbest = matchLoc[0]
    #   about y
    if matchLoc[1] - 2 >= 0 and matchLoc[1] + 2 < corrTm.shape[0]:
#        n = 5
        n = 3 # so far we only use n = 3 (taking 3 points, rather than 5 points)
        xmat = np.zeros((n,3), dtype=float)
        ymat = np.zeros((n,1), dtype=float)
        j = matchLoc[0]
        for i in range(n):
            xmat[i, 0] = (i - (n - 1) // 2) ** 2
            xmat[i, 1] = (i - (n - 1) // 2)
            xmat[i, 2] = 1.
            ymat[i, 0] = corrTm[(i - (n - 1) // 2) + matchLoc[1], j]
        amat = np.matmul(np.matmul(np.linalg.inv(np.matmul(xmat.transpose(), xmat)), xmat.transpose()), ymat)
        ybest = matchLoc[1] + amat[1,0] / (-2. * amat[0,0])
    elif matchLoc[1] - 1 >= 0 and matchLoc[1] + 1 < corrTm.shape[0]:
        n = 3
        xmat = np.zeros((n,3), dtype=float)
        ymat = np.zeros((n,1), dtype=float)
        j = matchLoc[0]
        for i in range(n):
            xmat[i, 0] = (i - (n - 1) // 2) ** 2
            xmat[i, 1] = (i - (n - 1) // 2)
            xmat[i, 2] = 1.
            ymat[i, 0] = corrTm[(i - (n - 1) // 2) + matchLoc[1], j]
        amat = np.matmul(np.matmul(np.linalg.inv(np.matmul(xmat.transpose(), xmat)), xmat.transpose()), ymat)
        ybest = matchLoc[1] + amat[1,0] / (-2. * amat[0,0])
    else:
        ybest = matchLoc[1]
    matchCenter= (xbest + wt // 2, ybest + ht // 2)
    cv.imshow("TM", corrTmImg); 

    # draw template
    ptsPrev = np.array((sx + wt / 2, sy + ht / 2))
    ptsNext = np.array((xbest + wt / 2, ybest + ht / 2))
    img3 = drawTemplateMatch(img1, img2, ptsPrev, ptsNext, (wt, ht))
    cv.imshow("Matches", img3); cv.waitKey(0); cv.destroyAllWindows()

def drawTemplateMatch(imgPrev, imgNext, ptsPrev, ptsNext, winSize):
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

# demoTemplateMatch()

