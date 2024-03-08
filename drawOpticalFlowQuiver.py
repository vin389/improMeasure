import cv2 as cv
import numpy as np

def drawOpticalFlowQuiver(
        prevImg,
        nextImg,
        prevPts,
        nextPts,
        winSize=None,
        arrowColor=(0,255,0),
        arrowThickness=1,
        arrowTipLength=0.1,
        winColor=(0,255,0),
        winThickness=1,
        shift=0):
    """
    

    Parameters
    ----------
    prevImg : TYPE
        DESCRIPTION.
    nextImg : TYPE
        DESCRIPTION.
    prevPts : TYPE
        DESCRIPTION.
    nextPts : TYPE
        DESCRIPTION.
    winSize : TYPE, optional
        DESCRIPTION. The default is None.
    arrowColor : TYPE, optional
        DESCRIPTION. The default is (0,255,0).
    arrowThickness : TYPE, optional
        DESCRIPTION. The default is 1.
    arrowTipLength : TYPE, optional
        DESCRIPTION. The default is 0.1.
    winColor : TYPE, optional
        DESCRIPTION. The default is (0,255,0).
    winThickness : TYPE, optional
        DESCRIPTION. The default is 1.
    shift : TYPE, optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    imgClone : TYPE
        DESCRIPTION.

    """
    #
    prevPts = prevPts.reshape(-1,2)
    nextPts = nextPts.reshape(-1,2)
    nPts = prevPts.shape[0]
    # check
    if prevPts.shape != nextPts.shape:
        print("# Error: drawOpticalFlowQuiver(): prevPts and nextPts must have the same size.")
        return
    # Convert winSize to a flattened numpy array
    if type(winSize) != type(None):
        if np.array(winSize).flatten().size == 1:
            winSize = (int(winSize + 0.5), int(winSize + 0.5))
        winSize = np.array(winSize).flatten()
    if type(arrowColor) == np.ndarray:
        arrowColor = (int(arrowColor[0]), int(arrowColor[1]), 
                      int(arrowColor[2]))
    if type(winColor) == np.ndarray:
        winColor = (int(winColor[0]), int(winColor[1]),
                    int(winColor[2]))
    # shift
    shiftFac = 2 ** shift
    # draw arrows and windows
    imgClone = nextImg.copy()
    for i in range(nPts):
        # draw an arrow
        cv.arrowedLine(
          imgClone, 
          (prevPts[i,:]*shiftFac + .5).astype('int'),
          (nextPts[i,:]*shiftFac + .5).astype('int'),
          arrowColor,
          thickness=arrowThickness,
          tipLength=arrowTipLength,
          shift=shift)
        # draw a window
        if type(winSize) != type(None):
            cv.rectangle(
              imgClone, 
              ((nextPts[i,:] - 0.5 * winSize) * shiftFac + .5).astype('int'),
              ((nextPts[i,:] + 0.5 * winSize) * shiftFac + .5).astype('int'),
              winColor,
              thickness=winThickness,
              shift=shift)
    # 
    return imgClone
    
    
    
if __name__ == '__main__':
    from mgridOnImage import mgridOnImage
    img = cv.imread('D:\\yuansen\\ImPro\\improMeasure\\examples\\2019rockfall\\P4RTK_1600_001.jpg')
    prevPts = mgridOnImage(img.shape[0], img.shape[1],
                         nHeight=img.shape[0] // 20,
                         nWidth=img.shape[1] // 20, dtype=np.float32)
    disps = np.array((10.,2.)).reshape(1,2)
    nextPts = prevPts + disps
    winSize = (11, 11)
    #
    drawOpticalFlowQuiver(
      img,
      img,
      prevPts,
      nextPts,
      winSize,
      arrowColor=(255,0,0),
      arrowThickness=2,
      arrowTipLength=0.5,
      shift=2
      )
    
    
    #
    cv.imshow('TEST', img)
    cv.waitKey(0)
    cv.destroyWindow('TEST')
    

    
