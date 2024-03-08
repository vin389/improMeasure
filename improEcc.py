#import numpy as np
import cv2 as cv
import numpy as np
import time

def improEcc(imgPrev: np.ndarray,
             imgNext: np.ndarray,
             ptsPrev: np.ndarray,
             ptsNext: np.ndarray,
             motionType: int=cv.MOTION_TRANSLATION,
             guess: int=0 # 0:given in ptsNext, 1:use template
             ):
    """
    Parameters
    ----------
    imgPrev : np.ndarray (imgH, imgW), dtype=np.uint8
        image before motion, supposed to be a full image, not just a template
    imgNext : np.ndarray (imgH, imgW), dtype=np.uint8
        image after motion
    ptsPrev : np.ndarray (nPoint, 6), dtype=np.float32
        one or more points with templates, in pixels, wrt imgPrev
        The 6 elements of each point are [x, y, rect_x, rect_y, rect_w, rect_h]
        Normally the x should be between rect_x and rect_x + rect_w, and the y 
        should be bewteen rect_y and rect_y + rect_h. 
    ptsNext : np.ndarray
        one or more points with templates, in pixels, wrt imgNext
        The 6 elements of each point are [x, y, rect_x, rect_y, rect_w, rect_h]
    motionType : int, optional
        DESCRIPTION. The default is cv.MOTION_TRANSLATION.
    guess : int, optional
        DESCRIPTION. The default is 0 # 0:given in ptsNext.
    1 : use template
        DESCRIPTION.

    Returns
    -------
    int
        DESCRIPTION.

    """
    
    return 0



def demoEcc():
    #
    # Define image sources
    fileImg1 = r'D:\yuansen\ImPro\improMeasure\examples\2022rcwall\leftPause\DSC02766.JPG'
    #fileImg2 = r'D:\yuansen\ImPro\improMeasure\examples\2022rcwall\leftPause\DSC02766.JPG'
    #fileImg2 = r'D:\yuansen\ImPro\improMeasure\examples\2022rcwall\leftPause\DSC02766.JPG'
    fileImg2 = r'D:\yuansen\ImPro\improMeasure\examples\2022rcwall\leftPause\DSC03183.JPG'
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
    # initial guess by template
    match_method = cv.TM_CCORR_NORMED
    corrTm = cv.matchTemplate(img2, tm, match_method)
    _minVal, _maxVal, minLoc, maxLoc = cv.minMaxLoc(corrTm, None)
    if (match_method == cv.TM_SQDIFF or match_method == cv.TM_SQDIFF_NORMED):
        matchLoc = minLoc
    else:
        matchLoc = maxLoc
    # ecc
    tic = time.time()
    warp_mode = cv.MOTION_HOMOGRAPHY # cv.MOTION_TRANSLATION, cv.MOTION_EUCLIDEAN, cv.MOTION_AFFINE, cv.MOTION_HOMOGRAPHY
    if warp_mode == cv.MOTION_HOMOGRAPHY :
        warp_matrix = np.eye(3, 3, dtype=np.float32)
        warp_matrix[0, 2] = matchLoc[0]
        warp_matrix[1, 2] = matchLoc[1]
    else:
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        warp_matrix[0, 2] = matchLoc[0]
        warp_matrix[1, 2] = matchLoc[1]
    number_of_iterations = 5000
    termination_eps = 1e-10
    criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)
    (cc, warp_matrix) = cv.findTransformECC (tm, img2, warp_matrix, warp_mode, criteria)
    toc = time.time()
    print("Ecc (%dx%d in %dx%d): %9.5f sec." %
          (img1.shape[1], img1.shape[0], img2.shape[1], img2.shape[0], toc - tic))
    # draw ECC
    tmRect = (sx, sy, wt, ht)
    img3 = drawEcc(img1, img2, tmRect, warp_matrix) 
    cv.imshow("Matches", img3); cv.waitKey(0); cv.destroyAllWindows()

def drawEcc(imgPrev, imgNext, tmRect, warp_matrix):
    imgShow = np.zeros((imgPrev.shape[0], 2*imgPrev.shape[1]), dtype=imgPrev.dtype)
    # background
    imgShow[:,0:imgPrev.shape[1]] = imgPrev
    imgShow[:,imgPrev.shape[1]:] = imgNext
    (sx, sy, wt, ht) = tmRect
    # tm of prev
    pt1 = (int(sx + .5), int(sy + .5))
    pt2 = (int(sx + wt + .5), int(sy + .5))
    pt3 = (int(sx + wt + .5), int(sy + ht + .5))
    pt4 = (int(sx + .5), int(sy + ht + .5))
    cv.line(imgShow, pt1, pt2, color=0, thickness=2)
    cv.line(imgShow, pt2, pt3, color=0, thickness=2)
    cv.line(imgShow, pt3, pt4, color=0, thickness=2)
    cv.line(imgShow, pt4, pt1, color=0, thickness=2)
    # tm of prev
    ptsp = np.array([
        [ 0,  0, 1],
        [wt,  0, 1],
        [wt, ht, 1],
        [ 0, ht, 1]], dtype=warp_matrix.dtype).transpose()
    ptsn = np.matmul(warp_matrix, ptsp)
    dx = np.array([imgPrev.shape[1] + .5,.5], dtype=ptsn.dtype).reshape(2,1).flatten()
    if ptsn.shape[0] == 2:
        pt1 = (ptsn[0:2,0].flatten() + dx).astype(int)
        pt2 = (ptsn[0:2,1].flatten() + dx).astype(int)
        pt3 = (ptsn[0:2,2].flatten() + dx).astype(int)
        pt4 = (ptsn[0:2,3].flatten() + dx).astype(int)
    else:
        pt1 = (ptsn[0:2,0].flatten() / ptsn[2,0] + dx).astype(int)
        pt2 = (ptsn[0:2,1].flatten() / ptsn[2,0] + dx).astype(int)
        pt3 = (ptsn[0:2,2].flatten() / ptsn[2,0] + dx).astype(int)
        pt4 = (ptsn[0:2,3].flatten() / ptsn[2,0] + dx).astype(int)
    cv.line(imgShow, pt1, pt2, color=0, thickness=2)
    cv.line(imgShow, pt2, pt3, color=0, thickness=2)
    cv.line(imgShow, pt3, pt4, color=0, thickness=2)
    cv.line(imgShow, pt4, pt1, color=0, thickness=2)
    return imgShow

# demoEcc()