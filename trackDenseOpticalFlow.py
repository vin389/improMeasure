import numpy as np
import cv2 as cv
import time

def trackDenseOpticalFlow(srcFiles: list,  # [file1, file2, ...]
                          pyr_scale=0.5,
                          levels: int =3,
                          winSize=11,
                          iterations=10,
                          poly_n=5,
                          poly_sigma=1.5,
                          flags=4, # cv.OPTFLOW_USE_INITIAL_FLOW = 4
                          flowOrDisp: str ='f', # 'f' or 'd' 
                          updatePrev=1, 
                          callbacks1=[],
                          callbacks2=[],
                          debug=False):
    """
    This function runs dense optical flow for multiple image files.
    
    Parameters
    ----------
    srcFiles : list of strings
        a list of strings. Each string is the (full path of) file name of an image.
        For example, ['c:/img/img000.jpg', 'c:/img/img001.jpg', 'c:/img/img002.jpg']
    pyr_scale : 
    levels : 
    winSize : 
    iterations : 
    poly_n : 
    poly_sigma : 
    flags : 
    flowOrDisp : str = 'f'
        the type of return data (either flow or displacement)
        'f' for flow (velocity, or relative displacement wrt each previous frame), 
        'd' for displacement (relative displ. wrt frame [0]) 
    updatePrev : int
        update the template (variable: prevImg) every several frames. 
        updatePrev == 0: the template image is always the first (index 0) image.
        updatePrev == 1: the template image is always the previous image. 
        updatePrev == 2: the template image is updated every other frame 
                         (for updatePrev = 2, the prev is [0], [2], [4], ...)
                         (for updatePrev = 5, the prev is [0], [5], [10], ...)
    callbacks1 : list , optional
        list of callback functions to execute after each frame of analysis.
        For example, if there are 10 images (10 files), callbacks1 will be 
        executed by 10 times.
        The default is [].
        Each callback function will be given 3 arguments:
            prevImg, nextImg, and flow
    callbacks2 : list , optional
        list of callback functions to execute after "all" frames of analysis. 
        Each callbacks2 will be executed only once.
        The default is [].
        Each callback function will be given 3 arguments:
            prevImg, nextImg, and flows or disps (of all frames)
    debug : bool = False
        print debug information on the screen

    Returns
    -------
        flow or displacement of each pixel 
        Format: numpy array that is (nFiles - 1, height, width, 2) dimensioned, 
        32-bit float numpy array. 
        The data is the flow (or displacement, depending on argument 
          flowOrDisp) of each pixel
        Warning: The array could be huge. For example:
            8000 frames of 1920x1080 --> array size: 124 GB (larger than main
                                         memory of many PCs)
            2000 frames of 800x600   --> array size: 7 GB 
             500 frames of 120x80    --> array size: 0.036 GB 
    """
    # set memory and initialize data
    nFiles = len(srcFiles)
    prevImg_i = 0
    prevImg = cv.imread(srcFiles[prevImg_i], cv.IMREAD_GRAYSCALE)
    if debug == True:
        print("# trackDenseOpticalFlow(): Reads image file %d." % prevImg_i) 
        
    nextImg_i = 1
    nextImg = cv.imread(srcFiles[nextImg_i], cv.IMREAD_GRAYSCALE)
    if debug == True:
        print("# trackDenseOpticalFlow(): Reads image file %d." % nextImg_i) 
    disps = np.zeros((nFiles - 1, prevImg.shape[0], prevImg.shape[1], 2), dtype=np.float32)
    flow = np.zeros((prevImg.shape[0], prevImg.shape[1], 2), dtype=np.float32)
    if debug:
        print("# trackDenseOpticalFlow(): allocated a big array sized: %.1f GBytes." 
              % (disps.size * disps.itemsize / (1024**3)))
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
            print("# trackDenseOpticalFlow(): Reset template frame to %d." % prev_i) 
        # set prevImg_i and prevImg
        if prev_i == nextImg_i:
            # if image prev_i has been in memory, use it, rather than reading from file.
            prevImg_i = nextImg_i
            prevImg = nextImg
            if debug == True:
                print("# trackDenseOpticalFlow(): prev_i case 1: prevImg frame %d." % prev_i) 
        elif prev_i == prevImg_i:
            # if image prev_i does not change, leave prevImg_i and prevImg unchanged.
            if debug == True:
                print("# trackDenseOpticalFlow(): prev_i case 2: prevImg frame %d." % prev_i) 
            pass # 
        else:
            # if image prev_i does not change, leave prevImg_i and prevImg unchanged.
            prevImg_i = prev_i
            prevImg = cv.imread(srcFiles[prevImg_i], cv.IMREAD_GRAYSCALE)
            if debug == True:
                print("# trackDenseOpticalFlow(): Read image file %d." % prevImg_i) 
            if debug == True:
                print("# trackDenseOpticalFlow(): prev_i case 3: prevImg frame %d." % prevImg_i) 
        # set nextImg_i and nextImg
        if nextImg_i != i:
            nextImg_i = i
            nextImg = cv.imread(srcFiles[nextImg_i], cv.IMREAD_GRAYSCALE)
            if debug == True:
                print("# trackDenseOpticalFlow(): Read image file %d." % nextImg_i) 
        # run dense optical flow
        tic = time.time()
        flow = cv.calcOpticalFlowFarneback(prevImg, nextImg, flow, pyr_scale, levels, winSize, iterations, poly_n, poly_sigma, flags)
        toc = time.time()
        if debug == True:
            print("# trackDenseOpticalFlow(): Dense opt_flow %d: %.3f s." % (i, toc - tic))
        # set disp (disp[i - 1] = disp[prevImg_i - 1] + flow)
        # The index is decreased by 1 because displacements of img0 is not stored,
        # and the displacements of img1 is storeed in disp[0]
        disps[i - 1] = disps[prevImg_i - 1] + flow
        # callbacks
        # for example: 
        #   trackDenseOpticalFlow(..., callbacks=[draw_flow()])
        for j in range(len(callbacks1)):
            tic = time.time()
            callbacks1[j](prevImg, nextImg, flow)
            toc = time.time()
            if debug == True:
                print("# trackDenseOpticalFlow(): frame %d callback %d: %.3f s." % (i, j, toc - tic))
    # convert disps to flow (if demanded)
    # The concept is: flows[i] = disps[i] - disps[i - 1]
    if flowOrDisp[0] == 'f':
        tic = time.time()
        for i in range(nFiles - 1, 0, -1):
            # for frame i (which data is stored at [i - 1])
            disps[i - 1] = disps[i - 1] - disps[i - 2]
        toc = time.time()
        if debug == True:
            print("# trackDenseOpticalFlow(): Converted from disp to flow (vel): %.3f s." % (toc - tic))
#
    # callbacks2
    # for example: 
    #   trackDenseOpticalFlow(..., callbacks2=[save_optflows_result()])
    for j in range(len(callbacks2)):
        tic = time.time()
        callbacks2[j](prevImg, nextImg, disps)
        toc = time.time()
        if debug == True:
            print("# trackDenseOpticalFlow(): frame %d callback2 %d: %.3f s." % (i, j, toc - tic))
#
    return disps
        
        # display flow field
#        showImg = draw_flow(nextImg, flow, step=8)
#        cv.imshow('flow', showImg)
#        ch = cv.waitKey(30)
#        if ch == 32 or ch == 27:
#            break
#    try:
#        cv.destroyWindow('flow')
#    except:
#        pass


def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    cv.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (_x2, _y2) in lines:
        cv.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis


def draw_flow_to_imshow(img, flow, step=16, winname='flow', delay=30):
    vis = draw_flow(img, flow, step)
    cv.imshow(winname, vis)
    ikey = cv.waitKey(delay)



# def trackDenseOpticalFlow_old(srcFiles=None, spec='L', updatePrev=0, dstFile=None):
#     """
#     This function runs dense optical flow. 
    
#     Parameters
#     ----------
#     srcFiles : TYPE, optional
#         DESCRIPTION. The default is None.
#     spec : str, optional
#         specification of the flow field, Eulerian or Lagrangian 
#         Lagrangian (spec[0] == 'L' or 'l'):  a way of looking at fluid motion 
#           where the observer follows an individual fluid parcel as it moves 
#           through space and time.
#         Eulerian (spec[0] == 'E' or 'e'): a way of looking at fluid motion 
#           that focuses on specific locations in the space through which the 
#           fluid flows as time passes. In Eulerian, the updatePrev is reset 
#           to 1 regardless what it is assigned by user. 
#         The default spec is 'L'         
#     updatePrev : int, optional
#         Updates template image (prev) every updatePrev 
#         If updatePrev == 0, it never updates templates. The prevImg would be
#             always the first image (i.e., srcFiles[0])
#         If updatePrev == 1, it updates templates every frame, i.e., templates
#             are the previous frame (i.e., srcFiles[i - 1].
#             If spec == 1, updatePrev will reset to 1 (and has to be 1)
#         If updatePrev == 5 (for example), it updates templates every 5 frames, 
#             i.e., for frames 1 to 5, it uses srcFiles[0], 
#                   for frames 6 to 10, it uses srcFiles[5].
#         The default updatePrev is 0. 
#     dstFile : TYPE, optional
#         DESCRIPTION. The default is None.
#     Returns
#     -------
#     None.
#     """
#     # Create file list
#     while True:
#         if type(srcFiles) == type(None):
#             print("# Enter (full-path) file names: ")
#             print("# For example: c:/test/IMG*.JPG or")
#             print("#              c:/test/IMG_%04d.JPG, [start], [number of files], [step]")
#             srcFiles = input2()
#         fileList = Filelist()
#         fileList.extendByString(srcFiles)
#         if fileList.nFiles() > 0:
#             break
#         print("# No file is added. Try again")
#         srcFiles == None
    
#     # set update period. 
#     nFiles = fileList.nFiles()
#     nextImg = np.empty(0) # to avoid parser warning 
#     for i in range(1, nFiles):
#         # read image
#         if i == 1:
#             prevImg = cv.imread(fileList.file(i - 1), cv.IMREAD_GRAYSCALE)
#             flow = np.zeros((prevImg.shape[0], prevImg.shape[1], 2), dtype=np.float32)
#         else:
#             prevImg = nextImg
#         nextImg = cv.imread(fileList.file(i), cv.IMREAD_GRAYSCALE)
#         # run dense optical flow
#         pyr_scale = 0.5
#         levels = 3
#         winSize = 11
#         iterations = 10
#         poly_n = 5
#         poly_sigma = 1.5
#         flags = cv.OPTFLOW_USE_INITIAL_FLOW
#         tic = time.time()
#         flow = cv.calcOpticalFlowFarneback(prevImg, nextImg, flow, pyr_scale, levels, winSize, iterations, poly_n, poly_sigma, flags)
#         toc = time.time()
#         print("Dense opt_flow %d: %.3f s." % (i, toc - tic))
#         # draw
#         cv.imshow('flow', draw_flow(nextImg, flow, 4))
#         ch = cv.waitKey(5)



if __name__ == '__main__':
#    srcFiles = r'D:\yuansen\ImPro\improMeasure\examples\2019rockfall\*.JPG'
#    trackDenseOpticalFlow_old(
#        srcFiles=r'D:\yuansen\ImPro\improMeasure\examples\2019rockfall\*.JPG', 
#        updatePrev=1, 
#        dstFile=None)
    import glob
    # 
    disps = trackDenseOpticalFlow(
      srcFiles=glob.glob(r'D:\yuansen\ImPro\improMeasure\examples\2019rockfall\*.JPG'),
      flowOrDisp='f', 
      updatePrev=1, 
      callbacks1=[lambda prevImg, nextImg, flow: draw_flow_to_imshow(
          nextImg, flow, step=8, winname="flow", delay=10)], 
      callbacks2=[],
      debug=True)
    cv.destroyWindow("flow")
    
    
    
    