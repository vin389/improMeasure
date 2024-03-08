import numpy as np
import cv2 as cv
import time


def trackOpticalFlowPyrLK(srcFiles: list, 
                          prevPts: np.ndarray,   
                          levels: int =3,
                          winSize=(11, 11),
                          iterations=10,
                          eps=0.01, 
                          flags=4, 
                          minEigThreshold=1e-4,
                          updatePrev=1,
                          posOrFlow='p', 
                          callbacks1=[],
                          callbacks2=[],
                          debug=False):
    """
    This function runs sparse optical flow (pyramid LK) on an image sequence. 

    Parameters
    ----------
    srcFiles : list
        list of image file names (full path).
    # [file1, file2, ...]                          
    prevPts : np.ndarray
        image points to track (defined in the first image: srcFiles[0]).
        The prevPts should be a N-by-2 32-bit float numpy array.
    levels : int, optional
        Maximum number of levels of the pyramid optical flow. The default is 3.
    winSize : tuple, optional
        window size for optical flow. The default is (11, 11).
    iterations : int, optional
        Maximum number of iterations. The default is 10.
    eps : double, optional
        The eps for optical flow iterations. The default is 0.01.
    flags : int , optional
        flags for optical flow. See OpenCV manual. The default is 
        cv.OPTFLOW_USE_INITIAL_FLOW (i.e., 4)
    minEigThreshold : float , optional
        the minimum eigen threshold. Only useful when flag 
        cv.OPTFLOW_LK_GET_MIN_EIGENVALS (8) is enabled. 
        See OpenCV manual. The default value is 1e-4. 
    updatePrev : int
        update the template (variable: prevImg) every several frames. 
        updatePrev == 0: the template image is always the first (index 0) image.
        updatePrev == 1: the template image is always the previous image. 
        updatePrev == 2: the template image is updated every other frame 
                         (for updatePrev = 2, the prev is [0], [2], [4], ...)
                         (for updatePrev = 5, the prev is [0], [5], [10], ...)
    posOrFlow : string (1-char string) : optional
        'p' for returning positions of each point of each frame
        'f' for returning flows of each point of each frame 
            flow is position[iFrame] - position[iFrame - 1]
    callbacks1 : list , optional
        list of callback functions to execute after each frame of analysis.
        For example, if there are 10 images (10 files), callbacks1 will be 
        executed by 10 times.
        The default is [].
        Each callback function will be given 6 arguments:
            i, prevImg, nextImg, prevPts, nextPts, and winSize
            (i is the frame index. First frame is index 0.)
            The prevPts and nextPts are positions, not flows. 
    callbacks2 : list , optional
        list of callback functions to execute after "all" frames of analysis. 
        Each callbacks2 will be executed only once.
        The default is [].
        Each callback function will be given 5 arguments:
            prevImg, nextImg, prevPts, nextPts, and winSize
    debug : bool , optional
        debug mode. The default is False.

    Returns
    -------
    Optical flow results, which are the image coordinates of tracking points.
    It is a 32-bit float numpy array, the dimension is (nFiles, nPoints, 2).   
    
    """
    # set memory and initialize data
    nFiles = len(srcFiles)
    prevPts = prevPts.reshape(-1,2)
    nPoints = prevPts.shape[0]
    pos = np.zeros((nFiles,nPoints, 2), dtype=np.float32)
    pos[0,:,:] = prevPts
    if debug:
        print("# trackOpticalFlowPyrLK(): allocated an array sized: %.1f MBytes." 
              % (pos.size * pos.itemsize / (1024**2)))
    # winSize
    if np.array(winSize).flatten().size == 1:
        winSize = (int(winSize + 0.5), int(winSize + 0.5))
    # read images
    prevImg_i = 0
    prevImg = cv.imread(srcFiles[prevImg_i], cv.IMREAD_GRAYSCALE)
    if debug == True:
        print("# trackOpticalFlowPyrLK(): Reads image file %d." % prevImg_i) 
    nextImg_i = 1
    nextImg = cv.imread(srcFiles[nextImg_i], cv.IMREAD_GRAYSCALE)
    if debug == True:
        print("# trackOpticalFlowPyrLK(): Reads image file %d." % nextImg_i) 
        
    # run loop doing optical flow analysis
    for i in range(1, nFiles):
        # calculate index of the prev image
        # prev_i is the "demanded" index of the prevImg.
        # prevImg_i is the "current" index of current prevImg  
        # prev_i could be different from prevImg_i. prevImg_i will later be set to prev_i
        if updatePrev == 0:
            prev_i = 0
        elif updatePrev == 1:
            prev_i = i - 1
        else:
            prev_i = ((i - 1) // updatePrev) * updatePrev
            print("# trackOpticalFlowPyrLK(): Reset template frame to %d." % prev_i) 
        # set prevImg_i and prevImg
        if prev_i == nextImg_i:
            # if image prev_i has been in memory, use it, 
            # and this program does not need to read image again 
            # from the file in order to save time 
            prevImg_i = nextImg_i
            prevImg = nextImg
            if debug == True:
                print("# trackOpticalFlowPyrLK(): prev_i case 1: prevImg frame %d." % prev_i) 
        elif prev_i == prevImg_i:
            # if image prev_i does not change, leave prevImg_i and prevImg unchanged.
            if debug == True:
                print("# trackOpticalFlowPyrLK(): prev_i case 2: prevImg frame %d." % prev_i) 
            pass # 
        else:
            # if image prev_i does not change, leave prevImg_i and prevImg unchanged.
            prevImg_i = prev_i
            prevImg = cv.imread(srcFiles[prevImg_i], cv.IMREAD_GRAYSCALE)
            if debug == True:
                print("# trackOpticalFlowPyrLK(): Read image file %d." % prevImg_i) 
            if debug == True:
                print("# trackOpticalFlowPyrLK(): prev_i case 3: prevImg frame %d." % prevImg_i) 
        # set nextImg_i and nextImg
        if nextImg_i != i:
            nextImg_i = i
            nextImg = cv.imread(srcFiles[nextImg_i], cv.IMREAD_GRAYSCALE)
            if debug == True:
                print("# trackOpticalFlowPyrLK(): Read image file %d." % nextImg_i) 
        # estimate nextPts
        nextPts = prevPts.copy()
        # run dense optical flow
        tic = time.time()
#        status=[]
#        err=[]
        nextPts, status, err = cv.calcOpticalFlowPyrLK(
            prevImg, 
            nextImg, 
            prevPts,  
            nextPts, # prediction
#            status=status,
#            err=err,
            winSize=winSize,
            maxLevel=levels,
            criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, iterations, eps),
            minEigThreshold=minEigThreshold)
        toc = time.time()
        if debug == True:
            print("# trackOpticalFlowPyrLK(): PyrLK opt_flow %d: %.3f s." % (i, toc - tic))
        # save one-frame data to all-frame data
        pos[i,:,:] = nextPts
        # callbacks 1 (for each frame)
        # for example: 
        #   trackDenseOpticalFlow(..., callbacks=[draw_flow()])
        for j in range(len(callbacks1)):
            tic = time.time()
            callbacks1[j](i, prevImg, nextImg, prevPts, nextPts, winSize)
            toc = time.time()
            if debug == True:
                print("# trackOpticalFlowPyrLK(): frame %d callbacks1 %d: %.3f s." % (i, j, toc - tic))
        # convert position to flow  
        if posOrFlow[0] == 'f':
            tic = time.time()
            for i in range(nFiles - 1, 0, -1):
                # for frame i (which data is stored at [i - 1])
                pos[i] = pos[i] - pos[i - 1]
            toc = time.time()
            if debug == True:
                print("# trackOpticalFlowPyrLK(): Converted from disp to flow (vel): %.3f s." % (toc - tic))
    # callbacks 2 (for all frames)
    for j in range(len(callbacks2)):
        tic = time.time()
        callbacks2[j](pos)
        toc = time.time()
        if debug == True:
            print("# trackOpticalFlowPyrLK(): callbacks2 %d: %.3f s." % (j, toc - tic))


    # return
    return pos

# callback functions 

def imshow_drawOpticalFlowQuiver_v0(i, prevImg, nextImg, prevPts, nextPts, 
                                 winSize):
#    from drawOpticalFlowQuiver import drawOpticalFlowQuiver
    imgShow = cv.cvtColor(nextImg, cv.COLOR_GRAY2BGR)
    imgShow = drawOpticalFlowQuiver(prevImg, imgShow, prevPts, nextPts, 
                winSize, shift=2)
    imgShow = cv.resize(imgShow, (-1, -1), fx=.5, fy=.5)
    cv.imshow('Flow1', imgShow)
    cv.waitKey(5)

def imshow_drawOpticalFlowColormap_v0(i, prevImg, nextImg, prevPts, nextPts, 
                                winSize):
    imgShow = cv.cvtColor(nextImg, cv.COLOR_GRAY2BGR)
    imgShow = drawOpticalFlowColormap(imgShow, prevPts, nextPts, winSize,
                cmap=cv.COLORMAP_BONE, 
                clim=(0,0),
                opacity=0.5)
    imgShow = cv.resize(imgShow, (-1, -1), fx=.5, fy=.5)
    cv.imshow('Flow2', imgShow)
    cv.waitKey(5)
    
    
def imshow_drawOpticalFlowQuiver_v1(i, prevImg, nextImg, prevPts, nextPts, 
      winSize, arrowColor, arrowThickness, arrowTipLengthRatio, 
      winColor, winThickness, shift, 
      imshowMaxSize):
    from drawOpticalFlowQuiver import drawOpticalFlowQuiver
    imgShow = cv.cvtColor(nextImg, cv.COLOR_GRAY2BGR)
    imgShow = drawOpticalFlowQuiver(prevImg, imgShow, prevPts, nextPts, 
      winSize, arrowColor, arrowThickness, arrowTipLengthRatio, 
      winColor, winThickness, shift)
    imgShow = cv.resize(imgShow, (-1, -1), fx=.5, fy=.5)
    resizeFact_X = imshowMaxSize[0] / imgShow.shape[1]
    resizeFact_Y = imshowMaxSize[1] / imgShow.shape[0]
    resizeFact = np.fmin(resizeFact_X, resizeFact_Y)
    imgShow = cv.resize(imgShow, (-1, -1), fx=resizeFact, fy=resizeFact)
    cv.imshow('FlowQuiver', imgShow)
    cv.waitKey(5)
    
    
def imshow_drawOpticalFlowColormap_v1(i, prevImg, nextImg, prevPts, nextPts, 
                                winSize, cmap, clim, opacity, imshowMaxSize):
    from drawOpticalFlowColormap import drawOpticalFlowColormap
    imgShow = cv.cvtColor(nextImg, cv.COLOR_GRAY2BGR)
    imgShow = drawOpticalFlowColormap(imgShow, prevPts, nextPts, winSize,
                cmap=cmap, 
                clim=clim,
                opacity=opacity)
    resizeFact_X = imshowMaxSize[0] / imgShow.shape[1]
    resizeFact_Y = imshowMaxSize[1] / imgShow.shape[0]
    resizeFact = np.fmin(resizeFact_X, resizeFact_Y)
    imgShow = cv.resize(imgShow, (-1, -1), fx=resizeFact, fy=resizeFact)
    cv.imshow('FlowColormap', imgShow)
    cv.waitKey(5)
    

def saveOpticalFlowOneFrame_v1(i, prevImg, nextImg, prevPts, nextPts, winSize,
                            filename):
    filename_i = filename % i
    if filename_i[-4:] == '.csv':
        np.savetxt(filename_i, nextPts.reshape(-1, 2), delimiter=' , ')
    if filename_i[-4:] == '.npy':
        np.save(filename_i, nextPts)
    if i == 1: 
        # if i == 1, run this function as if i = 0 
        filename_i = filename % 0
        if filename_i[-4:] == '.csv':
            np.savetxt(filename_i, prevPts.reshape(-1, 2), delimiter=' , ')
        if filename_i[-4:] == '.npy':
            np.save(filename_i, prevPts)
    
def saveOpticalFlowAllFrames_v1(pos, filename):
    if filename[-4:] == '.csv':
        np.savetxt(filename, pos.reshape(-1, 2), delimiter=' , ')
    if filename[-4:] == '.npy':
        np.save(filename, pos)
   

if __name__ == '__main__':
    import glob
    from mgridOnImage import mgridOnImage
    from drawOpticalFlowQuiver import drawOpticalFlowQuiver
    from drawOpticalFlowColormap import drawOpticalFlowColormap
       
    while (True):
        trackOpticalFlowPyrLK(
            # assign file sources
            srcFiles=glob.glob(
                r'D:\yuansen\ImPro\improMeasure\examples\2019rockfall\*.JPG'),
            # assign semi-dense grid for tracking
            prevPts=mgridOnImage(900, 1600, nHeight=90, nWidth=160, 
                                 dtype=np.float32),
            # optical flow parameters
            levels=3,
            winSize=11,
            iterations=10,
            eps=0.01,
            flags=cv.OPTFLOW_USE_INITIAL_FLOW,
            # update template period
            updatePrev=1,
            posOrFlow='p',
            # callback functions per step
            callbacks1=[imshow_drawOpticalFlowQuiver_v0, 
                        imshow_drawOpticalFlowColormap_v0],
            callbacks2=[],
            # 
            debug=True)
        ikey = cv.waitKey(5000)
#        break
        if ikey == 27 or ikey == 32:
            break
    cv.destroyWindow('Flow1')
    cv.destroyWindow('Flow2')