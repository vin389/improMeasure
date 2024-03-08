import numpy as np
import cv2 as cv
import glob
from inputs import input2
from npFromStr import npFromStr
from trackOpticalFlowPyrLK import trackOpticalFlowPyrLK
from trackOpticalFlowPyrLK import imshow_drawOpticalFlowColormap_v1
from trackOpticalFlowPyrLK import imshow_drawOpticalFlowQuiver_v1
from trackOpticalFlowPyrLK import saveOpticalFlowOneFrame_v1
from trackOpticalFlowPyrLK import saveOpticalFlowAllFrames_v1

#from drawOpticalFlowColormap import drawOpticalFlowColormap
#from drawOpticalFlowQuiver import drawOpticalFlowQuiver


def icf_opticalFlow(
        _files : str = None,
        _nPtsxy : str = None, 
        _maxLevel : str = None,
        _winSize : str = None,
        _nItr : str = None,
        _eps : str = None,
        _flags : str = None,
        _minEigThreshold : str = None,
        _updateTmplt : str = None,
        _posOrFlow : str = None,
        _showCmap : str = None,
        _showQuiver : str = None,
        _oFilesPath : str = None,
        _oFilePathAll : str = None, 
        _debug : str = None):
    """
    This function runs sparse optical flow (calcOpticalFlowPyrLK) for 
    multiple images. 
       You are supposed to provide the following information: 
    (1) Enter image files:
        The input will be processed by glob.glob. So your input should be able
        to be parsed by glob. Find glob for details.
        A single-char 'q' will exit this function.
        For example:
        c:\improMeasure\examples\2019rockfall\*.JPG
    (2) Enter numbers of tracking points:
        nX nY defines nX-by-nY tracking points over the image. It is like 
        virtually splits an image uniformly into nX by nY regions and 
        defines a tracking point at the center of each region.
        A negative value will exit this function.
        A single-char 'q' will exit this function.
        For example:
        120 60 
    (3) Enter maximum number of levels (default is 3):
        A non-positive input will set maximum level to 3.
        A single-char 'q' will exit this function.
        For example:
        3
    (4) Enter the window size of optical flow (default is 21 21)
        This defines nX pixels along x and nY pixels along y for the window.
        An input that is not greater than 3 will set window size to 3.
        A single-char 'q' will exit this function.
        For example:
        21 21
    (5) Enter maximum number of iterations (default is 30):
        A non-positive input will set iterations to 30.
        A single-char 'q' will exit this function.
        For example:
        30
    (6) Enter eps for iteration criteria (default is 0.01):
        A non-positive input will set iterations to 0.01.
        A single-char 'q' will exit this function.
        For example:
        0.01
    (7) Enter flags for optical flow (default is 4):
        cv.OPTFLOW_USE_INITIAL_FLOW: 4
        cv.OPTFLOW_LK_GET_MIN_EIGENVALS: 8
        See OpenCV manual for details.
        A negative input will set flags to 4.
        A single-char 'q' will exit this function.
        For example:
        4
    (8) Enter minEigThreshold for optical flow (default is 1e-4):
        A non-positive input will set minEigThreshold to 1e-4. 
        A single-char 'q' will exit this function.
        For example:
        1e-4
    (9) Enter update-template interval (default is 1):
        0: templates are always from the first frame ([0])
        1: templates are from the previous frame ([i-1])
        2: templates are rom frame [0] when analyzing frame [1] and [2], 
           templates are from frame [2] when analyzing frame [3] and [4], etc.
        3: templates are from frame [0] when analyzing frame [1] to [3],
           templates are from frame [3] when analyzing frame [4] to [6], etc.
        And so on. 
        A negative input will set interval to 1.
        A single-char 'q' will exit this function.
        For example:
        1
    (10) Enter type of return data ('p' for positions or 'f' for flows)
         Positions are the image coordinates of each tracking points.
         Flows are the movement of each tracking points relative to the 
         previous frame. For a frame i, flow[i] equals to 
         position[i] - position[i-1]
         A single-char 'q' will exit this function.
         For example:
         p
    (11) Showing colormap of the result (Colormap, minValue, maxValue, opacity,
         max imshow width, max imshow height). 
         If colormap is negative, now colormap will not show.
         A single-char 'q' will exit this function.         
         A single-integer -1 indicates not showing colormap.
         AUTUMN(0), BONE(1), JET(2), WINTER(3), RAINBOW(4), OCEAN(5), 
         SUMMER(6), SPRING(7), COOL(8), HSV(9), PINK(10), HOT(11), PARULA(12),
         MAGMA(13), INFERNO(14), PLASMA(15), VIRIDIS(16), CIVIDIS(17),
         TWILIGHT(18), TWILIGHT_SHIFTED(19), TURBO(20), DEEPGREEN(21)
         See OpenCV manual for details of colormap. 
         For example:
         5  0. 10. .5 800 450
         -1 (for not showing colormap)
    (12) Showing quiver of the result (arrow color, arrow thickness, arrow 
         tip length ratio, window box color, window box thickness, draw shift,
         max imshow width, max imshow height).
         If arrow color has negative value, quiver will not show.
         A single-char 'q' will exit this function. 
         -1 (for not showing colormap)
         For example:
         (0 255 0)   1  0.1  (0 255 0)  1  0  800 450
    (13) Enter file path to save data of each frame. Use C style (%d).
         The dimension of result of each frame will be (nY, nX, 2), where 
         nY and nX are number of tracking points along y and x. 
         The file format will be .csv or .npy.
         The end of input must be .csv or npy. Otherwise, data will 
         not be saved. 
         A single-char (except q), or input not ending with .csv or .npy.
         will not save data of each frame.
         A single-char 'q' will exit this function. 
         For example:
         c:\test\optflow_%06d.npy
    (14) Enter file path to save data of all frames. 
         The dimension of result of all frames will be (nFrames, nY, nX, 2),  
         where nFrames is number of frames (images), nY and nX are number of 
         tracking points along y and x. 
         The file format will be .csv or .npy.
         The end of input must be .csv or .npy. Otherwise, data will 
         not be saved. 
         A single-char (except q), or input not ending with .csv or .npy.
         will not save data of each frame.
         A single-char 'q' will exit this function. 
         For example:
         c:\test\optflow_all_frames.npy
    (15) Enable debug mode or not. (1: debug mode, 0: non-debug mode)
         For example:
         1

    Returns
    -------
    A numpy array with dimension (nFrames, nY, nX, 2), where nFrames is number 
    of frames (images), nY and nX are number of tracking points along y and x.

    """
    # image files (files)
    if type(_files) == type(None):
        print("# Enter image files:")
        print(r"#     E.g., c:\improMeasure\examples\2019rockfall\*.JPG")
        _files = input2()
    if _files == 'q':
        return 
    files = glob.glob(_files)
    nFiles = len(files)
    # print
    print("# Number of image files (nFiles): %d. They are:" % nFiles)
    for i in range(len(files)):
        print("#    %s" % files[i])
    
    # number of tracking points (nPtsxy)
    if type(_nPtsxy) == type(None):
        print("# Enter numbers of tracking points:")
        print(r"#    E.g., 120 60")
        _nPtsxy = input2()
    if _nPtsxy == 'q':
        return
    nPtsxy = npFromStr(_nPtsxy).astype(np.int32)
    if nPtsxy[0] < 0:
        return
    if nPtsxy[1] < 0:
        return
    # print
    print("# Numbers of tracking points (nPtsxy): %d %d" %
          (nPtsxy[0], nPtsxy[1]))
        
    # max number of levels (maxLevel)
    if type(_maxLevel) == type(None):
        print("# Enter maximum number of levels (maxLevel):")
        print(r"#    E.g., 3")
        _maxLevel = input2()
    if _maxLevel == 'q':
        return
    maxLevel = int(_maxLevel)
    if maxLevel <= 0:
        maxLevel = 3
    # print
    print("# Maximum number of levels (maxLevel): %d" % maxLevel)
        
    # window size (winSize)
    if type(_winSize) == type(None):
        print("# Enter window size of optical flow:")
        print("#    E.g., 21 21")
        _winSize = input2()
    if _winSize == 'q':
        return
    winSize = npFromStr(_winSize).astype(np.int32)
    if winSize.size == 1:
        winSize = np.array([winSize[0], winSize[0]], dtype=np.int32)
    if winSize[0] < 3:
        winSize[0] = 3
    if winSize[1] < 3:
        winSize[1] = 3
    # print
    print("# Window size (winSize): %d %d" % (winSize[0], winSize[1]))

    # max number of iterations (nItr)
    if type(_nItr) == type(None):
        print("# Enter maximum number of iterations (nItr):")
        print(r"#    E.g., 30")
        _nItr = input2()
    if _nItr == 'q':
        return
    nItr = int(_nItr)
    if nItr <= 0:
        nItr = 30
    # print
    print("# Maximum number of iterations (nItr): %d" % (nItr))

    # eps for optical flow (eps)
    if type(_eps) == type(None):
        print("# Enter eps for iteration criteria:")
        print(r"#    E.g., 0.01")
        _eps = input2()
    if _eps == 'q':
        return
    eps = float(_eps)
    if eps <= 0.:
        eps = .01
    # print
    print("# Eps for optical flow iteration (eps): %f" % eps)
    
    # flags for optical flow (flags)
    if type(_flags) == type(None):
        print("# Enter flags for optical flow:")
        print("#   (cv.OPTFLOW_USE_INITIAL_FLOW: 4")
        print("#   (cv.OPTFLOW_OK_GET_MIN_EIGENVALS: 8")
        print(r"#    E.g., 4")
        _flags = input2()
    if _flags == 'q':
        return
    flags = int(_flags)
    if flags <= 0.:
        flags = 4
    # print
    print("# Flags for optical flow (flags): %d" % flags)
    
    # minEigThreshold for optical flow (minEigThreshold)
    if type(_minEigThreshold) == type(None):
        print("# Enter minEigThreshold for optical flow:")
        print(r"#    E.g., 1e-4")
        _minEigThreshold = input2()
    if _minEigThreshold == 'q':
        return
    minEigThreshold = float(_minEigThreshold)
    if minEigThreshold <= 0.:
        minEigThreshold = 1e-4
    # print
    print("# minEigThreshold (minEigThreshold): %f" % minEigThreshold)
    
    # update-template interval (updateTmplt)
    if type(_updateTmplt) == type(None):
        print("# Enter update-template interval:")
        print(r"#    E.g., 1")
        _updateTmplt = input2()
    if _updateTmplt == 'q':
        return
    updateTmplt = int(_updateTmplt)
    if updateTmplt < 0:
        updateTmplt = 1
    # print
    print("# update-template interval (updateTmplt): %d" % updateTmplt)
    
    # position or flow (posOrFlow)
    if type(_posOrFlow) == type(None):
        print("# Enter type of return data ('p' for positions or 'f' for flows'):")
        print(r"#    E.g., p")
        _posOrFlow = input2()
    if _posOrFlow == 'q':
        return
    posOrFlow = _posOrFlow.strip()
    if posOrFlow[0] != 'f':
        posOrFlow = 'p'
    else:
        posOrFlow = 'f'
    # print
    print("# Position or flow: %s" % posOrFlow)
    
    # showing colormap (showCmap)
    if type(_showCmap) == type(None):
        print("# Showing colormap of the result (Colormap minValue maxValue opacity max_imshow_width max_imshow_height):")
        print(r"#    E.g., 5  0.  10.  .5  800  450")
        print(r"#    E.g., -1  (for not showing colormap)")
        _showCmap = input2()
    if _showCmap == 'q':
        return
    showCmapStrSplit = _showCmap.split()
    # showCmap
    showCmap_cmap = int(showCmapStrSplit[0])
    if int(showCmapStrSplit[0]) >= 0:
        showCmap_climMin = float(showCmapStrSplit[1])
        showCmap_climMax = float(showCmapStrSplit[2])
        showCmap_opacity = float(showCmapStrSplit[3])
        showCmap_imshowMaxWidth = int(showCmapStrSplit[4])
        showCmap_imshowMaxHeight = int(showCmapStrSplit[5])
    # print
    if showCmap_cmap >= 0:
        print("# Showing colormap (showCmap_cmap): %d" % showCmap_cmap)
        print("#   Clim of colormap (showCmap_climMin showCmap_climMax): %f %f" % (showCmap_climMin, showCmap_climMax))
        print("#   Opacity of colormap (showCmap_opacity): %f" % showCmap_opacity)
        print("#   Max imshow window size of colormap (showCmap_imshowMaxWidth showCmap_imshowMaxHeight): %d %d" % (showCmap_imshowMaxWidth, showCmap_imshowMaxHeight))
    else:
        print("# Not showing colormap")
    
    # showing quiver (showQuiver)
    if type(_showQuiver) == type(None):
        print("# Showing quiver of the result (allow_color arrow_thickness arrow_tip_ratio win_box_color win_box_thickness draw_shift max_imshow_width max_imshow_height):")
        print(r"#    E.g., (0 255 0)   1  0.1  (0 255 0)  1  0  800 450")
        print(r"#    E.g., -1  (for not showing quiver)")
        _showQuiver = input2()
    if _showQuiver == 'q':
        return
    showQuiverNp = npFromStr(_showQuiver)
    # showQuiver
    if showQuiverNp[0] < 0:
        showQuiver_arrowColor = np.array((-1,-1,-1), dtype=np.int32)
    else:
        showQuiver_arrowColor = showQuiverNp[0:3].astype(np.uint8)
        showQuiver_arrowThickness = int(showQuiverNp[3])
        showQuiver_tipLengthRaio = float(showQuiverNp[4])
        showQuiver_winBoxColor = showQuiverNp[5:8].astype(np.int32)
        showQuiver_winBoxThickness = int(showQuiverNp[8])
        showQuiver_shift = int(showQuiverNp[9])
        showQuiver_imshowMaxWidth = int(showQuiverNp[10])
        showQuiver_imshowMaxHeight = int(showQuiverNp[11])
    # print
    if showQuiver_arrowColor[0] >= 0:
        print("# Showing quiver color (showQuiver_arrowColor): %d %d %d" % 
              (showQuiver_arrowColor[0], showQuiver_arrowColor[1], 
               showQuiver_arrowColor[2]))
        print("#   quiver arrow thickness (showQuiver_arrowThickness): %d" % 
              showQuiver_arrowThickness)
        print("#   quiver arrow tip length ratio (showQuiver_tipLengthRaio): %f" % 
              showQuiver_tipLengthRaio)
        print("#   window box color (showQuiver_winBoxColor): %d %d %d" % 
              (showQuiver_winBoxColor[0], showQuiver_winBoxColor[1], 
               showQuiver_winBoxColor[2]))
        print("#   window box thickness (showQuiver_winBoxThickness): %d" % 
              showQuiver_winBoxThickness)
        print("#   drawing shift (showQuiver_shift): %d" % 
              showQuiver_shift)
        print("#   max imshow window size (showQuiver_imshowMaxWidth showQuiver_imshowMaxHeight): %d %d" %
              (showQuiver_imshowMaxWidth, showQuiver_imshowMaxHeight))
    else:
        print("# Not showing quiver")
    
    # file path to save data of each frame (oFilesPath)
    if type(_oFilesPath) == type(None):
        print("# Enter file path to save data of each frame:")
        print(r"#    E.g., c:\test\optflow_%06d.npy")
        print(r"#    E.g., . (for not saving result each frame)")
        _oFilesPath = input2()
    if _oFilesPath == 'q':
        return
    oFilesPath = _oFilesPath
    # print
    if len(oFilesPath) == 1:
        print("# Not saving data for each frame.")
    else:
        print("# Saving data for each frame at %s" % oFilesPath)
        
    # file path to save data of all frames (oFilePathAll)
    if type(_oFilePathAll) == type(None):
        print("# Enter file path to save data of all frames:")
        print(r"#    E.g., c:\test\optflow_all_frames.npy")
        print(r"#    E.g., . (for not saving result of all frames)")
        oFilePathAll = input2()
    if _oFilePathAll == 'q':
        return
    oFilePathAll = _oFilePathAll
    # print
    if len(oFilePathAll) == 1:
        print("# Not saving data for all frames.")
    else:
        print("# Saving data for all frames at %s" % oFilePathAll)

    # debug mode (debug)
    if type(_debug) == type(None):
        print("# Enable debug mode or not. (1: debug mode, 0: non-debug mode)")
        print(r"#    E.g., 1")
        _debug = input2()
    if _debug == 'q':
        return
    debug = int(_debug)
    # print
    if debug == 0:
        print("# Debug mode is disabled.")
    else:
        print("# Debug mode is enabled.")
#
# Run optical flow (by calling trackOpticalFlowPyrLK())
    from mgridOnImage import mgridOnImage
    # get image size (imgWidth, imgHeight)
    img0 = cv.imread(files[0])
    imgWidth = img0.shape[1]
    imgHeight = img0.shape[0]
    # prevPts
    prevPts = mgridOnImage(imgHeight, imgWidth,
                nHeight=nPtsxy[1], nWidth=nPtsxy[0], dtype=np.float32)
    # callbacks 1 (each frame during analysis)
    callbacks1 = []
    #   colormap callback
    if showCmap_cmap >= 0:
        cmap = showCmap_cmap
        clim = (showCmap_climMin, showCmap_climMax)
        opacity = showCmap_opacity
        imshowMaxSize = (showCmap_imshowMaxWidth, showCmap_imshowMaxHeight)
        func1 = lambda i, prevImg, nextImg, prevPts, nextPts, winSize:\
            imshow_drawOpticalFlowColormap_v1(\
                i, prevImg, nextImg, prevPts, nextPts, winSize,\
                cmap, clim, opacity, imshowMaxSize)
        callbacks1.append(func1)
    #   quiver callback
    if showQuiver_arrowColor[0] >= 0:
        func2 = lambda i, prevImg, nextImg, prevPts, nextPts, winSize:\
            imshow_drawOpticalFlowQuiver_v1(\
                i, prevImg, nextImg, prevPts, nextPts, winSize,\
                showQuiver_arrowColor, showQuiver_arrowThickness,\
                showQuiver_tipLengthRaio, showQuiver_winBoxColor,\
                showQuiver_winBoxThickness, showQuiver_shift, 
                (showQuiver_imshowMaxWidth, showQuiver_imshowMaxHeight))
        callbacks1.append(func2)
    # save file each frame
    if len(oFilesPath) > 1:
        func3 = lambda i, prevImg, nextImg, prevPts, nextPts, winSize:\
            saveOpticalFlowOneFrame_v1(\
                i, prevImg, nextImg, prevPts, nextPts, winSize,\
                oFilesPath)
        callbacks1.append(func3)
    # callbacks 2 (all frames after analysis)
    callbacks2 = []
    # save file for all frames
    if len(oFilePathAll) > 1:
        func4 = lambda pos: saveOpticalFlowAllFrames_v1(pos, oFilePathAll)
        callbacks2.append(func4)
    
    # run optical flow
    trackOpticalFlowPyrLK(
        srcFiles=files,
        prevPts=prevPts,
        levels=maxLevel, 
        winSize=winSize,
        iterations=nItr,
        eps=eps,
        flags=flags,
        updatePrev=updateTmplt, 
        posOrFlow=posOrFlow,
        callbacks1=callbacks1,
        callbacks2=callbacks2,
        debug=debug
        )
    
    pass
    return

if __name__ == '__main__':
    while(True):
        icf_opticalFlow(
            r"D:\yuansen\ImPro\improMeasure\examples\2019rockfall\*.JPG",
            "120 60", 
            "3",
            "15 15",
            "30",
            "1e-2",
            "4", 
            "1e-4", 
            "1",
            "p",
            "5  0. 10. .5 800 450 ",
            " (0 255 0)   1  0.1  (0 255 0)  1  0  800 450",
            r"D:\yuansen\ImPro\improMeasure\examples\2019rockfall\optflow_%06d.npy",
            r"D:\yuansen\ImPro\improMeasure\examples\2019rockfall\optflow_all_frames.npy",
            "1"
        )
        ikey = cv.waitKey(1000)
        if ikey == 32 or ikey == 27 or ikey == 'q':
            break
        break
    cv.destroyAllWindows()