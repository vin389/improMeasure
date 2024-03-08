import cv2 as cv
import numpy as np

def drawOpticalFlowColormap(
        img,
        prevPts,
        nextPts,
        winSize=None,
        cmap=cv.COLORMAP_JET,
        clim=(0,0),
        opacity=0.5):
    """
    

    Parameters
    ----------
    img : TYPE
        DESCRIPTION.
    prevPts : TYPE
        DESCRIPTION.
    nextPts : TYPE
        DESCRIPTION.
    winSize : TYPE, optional
        DESCRIPTION. The default is None.
    cmap : TYPE, optional
        DESCRIPTION. The default is cv.COLORMAP_JET.
    clim : TYPE, optional
        DESCRIPTION. The default is (0,0).
    opacity : TYPE, optional
        DESCRIPTION. The default is 0.5.

    Returns
    -------
    imgCmap : TYPE
        DESCRIPTION.

    """
    #
    prevPts = prevPts.reshape(-1,2)
    nextPts = nextPts.reshape(-1,2)
    nPts = prevPts.shape[0]
    # check
    if prevPts.shape != nextPts.shape:
        print("# Error: drawOpticalFlowColormap(): prevPts and nextPts must have the same size.")
        return
    # Convert winSize to a flattened numpy array
    if type(winSize) != type(None):
        if np.array(winSize).flatten().size == 1:
            winSize = (int(winSize + 0.5), int(winSize + 0.5))
        winSize = np.array(winSize).flatten()
    # allocate array for norms of flows
    normsFlow = np.zeros((nPts), dtype=np.float32)
    for i in range(nPts):
        normsFlow[i] = np.linalg.norm(nextPts[i] - prevPts[i])
    # check clim. Refine it if it is necessary.
    if clim[0] >= clim[1]:
        clim = (np.min(normsFlow), np.max(normsFlow))
        if clim[1] - clim[0] < 1e-12:
            clim = (clim[0] - 1e-12, clim[1] + 1e-12)
    # convert normsFlow to u255Flow (range 0-255)
    u255Flow = np.zeros((nPts), dtype=np.uint8)
    if np.min(normsFlow) < clim[0] or np.max(normsFlow) > clim[1]:
        # considering outliers, clim can be range of inliers.
        # In this case, we leave 0 and 255 for outliers           
        for i in range(nPts):
            if normsFlow[i] > clim[1]:
                u255Flow = 255
            elif normsFlow[i] < clim[0]:
                u255Flow = 0
            else:
                u255Flow[i] = int((normsFlow[i] - clim[0]) * 253 / (clim[1] - clim[0]) + 1.5)
    else:
        # Not considering outliers. clim is the range of all data.
        for i in range(nPts):
            u255Flow[i] = int((normsFlow[i] - clim[0]) * 255 / (clim[1] - clim[0]) + .5)
    # colormap
    u256 = np.zeros((256), dtype=np.uint8)
    for i in range(256):
        u256[i] = i
    colormap = cv.applyColorMap(u256, cmap).reshape(256,3)
    # draw color block. Block center is at prevPts[i]
    if img.shape[0] * img.shape[1] == img.size:
        imgCmap = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    else:
        imgCmap = img.copy()
    # draw color on imgCmap
    for i in range(nPts):
        # calculate rectangle corners
        pt1 = (prevPts[i] - 0.5 * winSize + 0.5).astype(int)
        pt2 = (prevPts[i] + 0.5 * winSize + 0.5).astype(int)
        # check corner bounds
        if pt1[0] < 0:
            pt1[0] = 0
        if pt1[0] >= imgCmap.shape[1]:
            pt1[0] = imgCmap.shape[1] - 1
        if pt2[0] < 0:
            pt2[0] = 0
        if pt2[0] >= imgCmap.shape[1]:
            pt2[0] = imgCmap.shape[1] - 1
        if pt1[1] < 0:
            pt1[1] = 0
        if pt1[1] >= imgCmap.shape[0]:
            pt1[1] = imgCmap.shape[0] - 1
        if pt2[1] < 0:
            pt2[1] = 0
        if pt2[1] >= imgCmap.shape[0]:
            pt2[1] = imgCmap.shape[0] - 1
        # create a small patch
        patch = np.zeros((pt2[1] - pt1[1], pt2[0] - pt1[0], 3), dtype=np.uint8)
        patch[:,:,:] = colormap[u255Flow[i]]
        # 
        imgCmap[pt1[1]:pt2[1], pt1[0]:pt2[0], :] =\
            (imgCmap[pt1[1]:pt2[1], pt1[0]:pt2[0], :] * (1. - opacity) + patch * opacity).astype(np.uint8)
    # return     
    return imgCmap
    
if __name__ == '__main__':
    from mgridOnImage import mgridOnImage
    img = cv.imread('D:\\yuansen\\ImPro\\improMeasure\\examples\\2019rockfall\\P4RTK_1600_001.jpg')
    prevPts = mgridOnImage(img.shape[0], img.shape[1],
                         nHeight=img.shape[0] // 20,
                         nWidth=img.shape[1] // 20, dtype=np.float32)
    disps = np.array((10.,2.)).reshape(1,2)
    nextPts = prevPts + disps
    winSize = (11, 11)
    prevPts = prevPts.reshape(-1,2)
    nextPts = nextPts.reshape(-1,2)
    #
    img = drawOpticalFlowColor(
      img, 
      prevPts,
      nextPts,
      winSize=21, 
      opacity=0.5)
    
    #
    cv.imshow('TEST', img)
    cv.waitKey(0)
    cv.destroyWindow('TEST')
    

    
