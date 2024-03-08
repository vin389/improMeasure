import os
import glob
import numpy as np
import time
from numba import njit

import cv2 as cv
from inputs import input2


def cvColorMapStr():
    return ['AUTUMN', 'BONE', 'JET', 'WINTER', 'RAINBOW', 'OCEAN', 'SUMMER', 
            'SPRING', 'COOL', 'HSV', 'PINK', 'HOT', 'PARULA', 'MAGMA', 
            'INFERNO', 'PLASMA', 'VIRIDIS', 'CIVIDIS', 'TWILIGHT', 
            'TWILIGHT_SHIFTED', 'TURBO', 'DEEPGREEN']


def coordsAfterResizing(oldXi, oldYi, factorX, factorY):
    newXi = (oldXi + 0.5) * factorX - 0.5
    newYi = (oldYi + 0.5) * factorY - 0.5
    return (newXi, newYi)

# internal function for numba
@njit
def calcPolyAndColor(field, posi_exp, cOut, clim, colorbar, resizeFact, i, j):
    ii = i + 1
    jj = j + 1
    fValue = field[i, j]
    if cOut[0] >= 0 and (fValue > clim[1] or fValue < clim[0]):
        fColor = (int(cOut[0]), int(cOut[1]), int(cOut[2]))
    else:
        fiColor = round(((fValue - clim[0]) / (clim[1] - clim[0])) * 255)
        fiColor = min(255, max(0, fiColor))
#        fColor = colorbar[fiColor].flatten().astype(np.uint8)
        fColor = (int(colorbar[fiColor,0,0]), int(colorbar[fiColor,0,1]), int(colorbar[fiColor,0,2]))
    poly_pts = np.zeros((4, 2), dtype=np.float32)
    poly_pts[0] = .5 * (posi_exp[ii,jj] + posi_exp[ii-1,jj-1])
    poly_pts[1] = .5 * (posi_exp[ii,jj] + posi_exp[ii+1,jj-1])
    poly_pts[2] = .5 * (posi_exp[ii,jj] + posi_exp[ii+1,jj+1])
    poly_pts[3] = .5 * (posi_exp[ii,jj] + posi_exp[ii-1,jj+1])
    poly_pts = (poly_pts + 0.5) * resizeFact - 0.5
    poly_pts = poly_pts + 0.5
    poly_pts = poly_pts.astype(np.int32)
    fColor = (int(fColor[0]), int(fColor[1]), int(fColor[2]))
    return poly_pts, fColor




def drawField(_bgImgFile=None,
              _fieldFile=None,
              _posiFile=None,
              _saveImgFile=None,
              _colormap=None,
              _reverseMap=None,
              _cOut=None,
              _clim=None,
              _viewWidth=None):
    # background image file
    if type(_bgImgFile) == type(None):
        print("# Enter background image file:")
        _bgImgFile = input2()
    bgImg_file = _bgImgFile
        
    # field file (a 2D ny by nx CSV numpy file which contains values of the field)
    if type(_fieldFile) == type(None):
        print("# Enter field file which contains values of fields (ny by nx, 2D numpy CSV format):")
        _fieldFile = input2()
    fieldFile = _fieldFile
        
    # position file (a 2D ny by 2nx CSV numpy file which constains image coordinates of each point in the field)  
    if type(_posiFile) == type(None):
        print("# Enter position file which contains image coordinates of each point in the field (ny by 2nx, 2D numpy CSV format):")
        _posiFile = input2()
    posiFile = _posiFile
        
    # save file
    if type(_saveImgFile) == type(None):
        print("# Enter image file that you want to save the plotted image:")
        _saveImgFile = input2()
    saveImgFile = _saveImgFile
        
    # colormap
    if type(_colormap) == type(None):
        print("# Enter colormap (0 ~ 21. 1:bone(gray:black->white), 2:jet(blue->red). See OpenCV manual for details): ")
        print("#   You can also enter colorbar name, e.g., JET")
        for i in range(22):
            print("#   %d:%s" % (i, cvColorMapStr()[i]))
        _colormap = input2()
    try:
        colormap = int(_colormap)
    except:
        colormap = -1
        for i in range(22):
            if _colormap.upper().find(cvColorMapStr()[i]) >= 0:
                colormap = i
                break

    # reverse colormap
    if type(_reverseMap) == type(None):
        print("# Do you want to reverse the colormap (0: No. 1: Yes.)?")
        _reverseMap = input2()
    reverseMap = int(_reverseMap) 
        
    # color of outliers (cOut)
    if type(_cOut) == type(None):
        print("# Enter the color of outlier (e.g., 255 255 255 for white): ")
        print("#   or -1 -1 -1 if you do not want to handle outlier.")
        _cOut = input2()
    cOut = _cOut
    cOut = cOut.replace('[',' ').replace(']',' ').replace(',',' ').replace(';',' ')
    cOut = cOut.replace('(',' ').replace(')',' ').replace('{',' ').replace('}',' ')
    cOut = (int(cOut.split()[0]), int(cOut.split()[1]), int(cOut.split()[2]))
        
    # lower and upper bound (clim)
    if type(_clim) == type(None):
        print("# Enter the lower and upper bounds (e.g., -1e-3  1e-3): ")
        print("# If you enter 0 0, they will be set to the min/max values of the field.")
        print("# If you enter two equal values, lower/upper bound will set based on percentile.")
        print("#   E.g., 5 5 will set lower/upper bounds to 5-th/95-th percentile.")
        _clim = input2()
    clim = _clim
    clim = clim.replace('[',' ').replace(']',' ').replace(',',' ').replace(';',' ')
    clim = clim.replace('(',' ').replace(')',' ').replace('{',' ').replace('}',' ')
    clim = (float(_clim.split()[0]), float(_clim.split()[1]))

    # image width of output image 
    if type(_viewWidth) == type(None):
        print("# Enter the width of output image (e.g., 400")
        print("# If you enter 0 or a negative value, the width will be set to 400.")
        print("# A greater value will generates large images and takes longer computing time for drawing.")
        _viewWidth = input2()
    viewWidth = int(_viewWidth)
    if viewWidth <= 0:
        viewWidth = 400
    
    # read background image (bgImg_file --> bgImg)
    bgImg = cv.imread(bgImg_file)
    
    # read field data (fieldFile --> field, nx, ny)
    # field is ny by nx, float32
    field = np.loadtxt(fieldFile, delimiter=',', dtype=np.float32)
    nx = field.shape[1]
    ny = field.shape[0]
    
    # check clim
    if clim[0] == 0 and clim[1] == 0:
        clim[0] = np.min(field.flatten())
        clim[1] = np.max(field.flatten())
    elif clim[0] == clim[1] and clim[0] < 0:
        clim[0] = np.min(field.flatten())
        clim[1] = np.max(field.flatten())
    elif clim[0] == clim[1]:
        the_percentile = clim[0]
        clim[0] = np.percentile(field.flatten(), the_percentile)
        clim[1] = np.percentile(field.flatten(), 100 - the_percentile)
    if clim[0] > clim[1]:
        clim = (clim[1], clim[0])
    if clim[0] == clim[1]:
        clim = (clim[0] - 1e-30, clim[0] + 1e-30)
    
    # read position (image coordinates) of field
    # (posiFile --> posi) 
    # posi is ny by nx by 2, float32
    posi = np.loadtxt(posiFile, delimiter=',', dtype=np.float32)
    posi = posi.reshape(ny, nx, 2)
    
    # create an expanded positions (posi -> posi_exp)
    # posi_exp is ny+2 by nx + 2 by 2, float32
    posi_exp = np.zeros((ny + 2, nx + 2, 2), dtype=np.float32)
    posi_exp[1:-1,1:-1,:] = posi
    posi_exp[ 0,1:-1,:] = 2 * posi[ 0,:,:] - posi[ 1,:,:]
    posi_exp[-1,1:-1,:] = 2 * posi[-1,:,:] - posi[-2,:,:]
    posi_exp[1:-1, 0,:] = 2 * posi[:, 0,:] - posi[:, 1,:]
    posi_exp[1:-1,-1,:] = 2 * posi[:,-1,:] - posi[:,-2,:]
    posi_exp[ 0, 0,:] = 2 * posi[ 0, 0,:] - posi[ 1, 1,:]
    posi_exp[ 0,-1,:] = 2 * posi[ 0,-1,:] - posi[ 1,-2,:]
    posi_exp[-1, 0,:] = 2 * posi[-1, 0,:] - posi[-2, 1,:]
    posi_exp[-1,-1,:] = 2 * posi[-1,-1,:] - posi[-2,-2,:]

    # create colorbar (colormap --> colorbar)
    # colorbar is a 256 by 1, 3-channel image
    if reverseMap == True:
        # reverse the colormap
        colorbargray = np.array(range(255,-1,-1), dtype=np.uint8).reshape(256,1)
    else:
        # keep the original colormap
        colorbargray = np.array(range(256), dtype=np.uint8).reshape(256,1)
    colorbar = cv.applyColorMap(colorbargray, colormap)
       
    # plot color on background image (bgImg --> plottedImg)
    # plottedImg has a 100-pixel vertical expansion for colorbar and clim 
    resizeFact = viewWidth * 1.0 / bgImg.shape[1]
    viewHeight = int(viewWidth * bgImg.shape[0] / bgImg.shape[1]) + 100 
    plottedImg = np.zeros((viewHeight, viewWidth, 3), dtype=np.uint8)
    bgImgResized = cv.resize(bgImg, dsize=(viewWidth, viewHeight-100))
    plottedImg[0:bgImgResized.shape[0], :, :] = bgImgResized
    for i in range(ny):
        for j in range(nx):
            useNumba = True
            if useNumba == False:
                fValue = field[i,j]
                if cOut[0] >= 0 and (fValue > clim[1] or fValue < clim[0]):
                    fColor = cOut
                else:
                    fiColor = round(((fValue - clim[0]) / (clim[1] - clim[0])) * 255)
                    fiColor = min(255, max(0, fiColor))
                    fColor = colorbar[fiColor].flatten().astype(np.uint8)
                poly_pts = np.zeros((4, 2), dtype=np.float32)
                ii = i + 1
                jj = j + 1
                poly_pts[0] = .5 * (posi_exp[ii,jj] + posi_exp[ii-1,jj-1])
                poly_pts[1] = .5 * (posi_exp[ii,jj] + posi_exp[ii+1,jj-1])
                poly_pts[2] = .5 * (posi_exp[ii,jj] + posi_exp[ii+1,jj+1])
                poly_pts[3] = .5 * (posi_exp[ii,jj] + posi_exp[ii-1,jj+1])
                poly_pts_resized = (poly_pts + 0.5) * resizeFact - 0.5
                poly_pts_resized = np.round(poly_pts_resized).astype(np.int32)
                fColor = (int(fColor[0]), int(fColor[1]), int(fColor[2]))
            else:
                poly_pts_resized, fColor = calcPolyAndColor(field, posi_exp, cOut, clim, colorbar, resizeFact, i, j)
#            def calcPolyAndColor(field, posi_exp, cOut, clim, colorbar, resizeFact, ii, jj):

            plottedImg = cv.fillPoly(plottedImg, pts=[poly_pts_resized], color=fColor)          
    # plot colorbar
    cbarX0 = 3
    cbarY0 = 3 + bgImgResized.shape[0]
    cbarW = plottedImg.shape[1] - 2 * cbarX0
    cbarH = 22
    for i in range(cbarW):
        ic = round((i * 255) / (cbarW - 1))
        plottedImg[cbarY0:(cbarY0+cbarH+1), cbarX0 + i, :] = colorbar[ic, 0, :]
    # plot clim text    
    text = 'Colormap range: %f / %f' % (clim[0], clim[1])
    plottedImg = cv.putText(plottedImg, text, 
               org=(3, plottedImg.shape[0] - 15),
               fontFace=cv.FONT_HERSHEY_SIMPLEX,
               fontScale=0.5,
               thickness=1,
               color=(255,255,255))
    
    
#    from imshow2 import imshow2
#    imshow2("Field", plottedImg)

    # save file (saveImgFile)
    saveImgFile_dir = os.path.dirname(saveImgFile)
    if os.path.exists(saveImgFile_dir) == False:
        os.makedirs(saveImgFile_dir)
#    plottedImgResized = cv.resize(plottedImg, dsize=(0,0), fx=0.25, fy=0.25)
    cv.imwrite(saveImgFile, plottedImg)

def test1():
    bgImgFile = r'D:\yuansen\ImPro\improMeasure\examples\2022rcwall\rectf\rectf_c1_step4.bmp'
    fieldFile = r'D:\yuansen\ImPro\improMeasure\examples\2022rcwall\fields\Step_4_Cam_1_gxy_cellSize_30.csv'
    posiFile = r'D:\yuansen\ImPro\improMeasure\examples\2022rcwall\fields\Step_4_Cam_1_imgPts_cellSize_30.csv'
    saveImgFile = r'D:\yuansen\ImPro\improMeasure\examples\2022rcwall\fields\Step_4_Cam_1_ux_cellSize_30.JPG'
    colormap = 'jet'
    reverseMap = '0'
    cOut = ' 255  255  255'
    clim = '-5e-3 5e-3'
    viewWidth = '-1'
    drawField(_bgImgFile=bgImgFile,
              _fieldFile=fieldFile,
              _posiFile=posiFile,
              _saveImgFile=saveImgFile,
              _colormap=colormap,
              _reverseMap=reverseMap,
              _cOut=cOut,
              _clim=clim,
              _viewWidth=viewWidth)
    
def test2():
    bgImgFile = r'D:\ExpDataSamples\20220200_NcreeNorth_RcWall\Analysis_improMeasure_Spc_1_Cams1_2\rectf\rectf_c1_step10.bmp'
    fieldFile = r'D:\ExpDataSamples\20220200_NcreeNorth_RcWall\Analysis_improMeasure_Spc_1_Cams1_2\fields\Step_10_Cam_1_exx_cellSize_15.csv'
    posiFile =  r'D:\ExpDataSamples\20220200_NcreeNorth_RcWall\Analysis_improMeasure_Spc_1_Cams1_2\fields\Step_10_Cam_1_imgPts_cellSize_15.csv'
    saveImgFile = r'D:\ExpDataSamples\20220200_NcreeNorth_RcWall\Analysis_improMeasure_Spc_1_Cams1_2\fields\Step_10_Cam_1_exx_cellSize_15.JPG'
    colormap = 'jet'
    reverseMap = '0'
    cOut = ' 255  255  255'
    clim = '-5e-2 5e-2'
    viewWidth = '-1'
    drawField(_bgImgFile=bgImgFile,
              _fieldFile=fieldFile,
              _posiFile=posiFile,
              _saveImgFile=saveImgFile,
              _colormap=colormap,
              _reverseMap=reverseMap,
              _cOut=cOut,
              _clim=clim,
              _viewWidth=viewWidth)
    
def test3():
    bgImgFile = r'D:\yuansen\ImPro\improMeasure\examples\2022rcwall_cracktest\rectf\rectf_c1_step2.bmp'
    fieldFile = r'D:\yuansen\ImPro\improMeasure\examples\2022rcwall_cracktest\fields\Step_2_Cam_1_crackOpening_cellSize_15.csv'
    posiFile =  r'D:\yuansen\ImPro\improMeasure\examples\2022rcwall_cracktest\fields\Step_2_Cam_1_imgPts_cellSize_15.csv'
    saveImgFile = r'D:\yuansen\ImPro\improMeasure\examples\2022rcwall_cracktest\fields\Step_2_Cam_1_crackOpening_cellSize_15.JPG'
    colormap = 'bone'
    reverseMap = '1'
    cOut = ' 255  255  255'
    clim = '0 1'
    viewWidth = '-1'
    drawField(_bgImgFile=bgImgFile,
              _fieldFile=fieldFile,
              _posiFile=posiFile,
              _saveImgFile=saveImgFile,
              _colormap=colormap,
              _reverseMap=reverseMap,
              _cOut=cOut,
              _clim=clim,
              _viewWidth=viewWidth
              )
    
def test_rcwalls():
    
    for ispc in range(4):
        _wdir = r'D:\ExpDataSamples\20220200_NcreeNorth_RcWall\Analysis_improMeasure_Spc_%d_Cams1_2' % (ispc + 1)
        wdir_rectf = os.path.join(_wdir, 'rectf')
        rectf_files = glob.glob(os.path.join(wdir_rectf) + '/*.*')
        nstep = len(rectf_files) // 2  # rectf_files has left and right cameras, and in this case we only do camera 0
        icam = 0 # if you want to run both cameras, you need to add a loop like, for icam in range(2):
        for istep in range(nstep):
            tic = time.time()
            bgImgFile = os.path.join(_wdir, 'rectf\Step_%d_Cam_%d_rectf.bmp' % (istep+1, icam+1))
        
            field='crack'
            csize = 15
            colormap = 'bone'
            reverseMap = '1'
            cOut = ' -1 -1 -1'
            clim = '0 1'
            viewWidth = '-1'
            posiFile =  os.path.join(_wdir, 'fields\Step_%d_Cam_%d_imgPts_cellSize_%s.csv' % (istep+1,icam+1,csize))
            fieldFile = os.path.join(_wdir, 'fields\Step_%d_Cam_%d_%s_cellSize_%d.csv' % (istep+1,icam+1,field,csize))
            clfilestr = '10'
            saveImgFile = os.path.join(_wdir, 'fields\Cam_%d_%s_cellSize_%d_cl%s' % (icam+1,field,csize,clfilestr),
                                       'Step_%d_Cam_%d_%s_cellSize_%d_cl%s.JPG' % (istep+1,icam+1,field,csize,clfilestr))
            drawField(_bgImgFile=bgImgFile,
                      _fieldFile=fieldFile,
                      _posiFile=posiFile,
                      _saveImgFile=saveImgFile,
                      _colormap=colormap,
                      _reverseMap=reverseMap,
                      _cOut=cOut,
                      _clim=clim,
                      _viewWidth=viewWidth)


            field='gxy'
            csize = 60
            colormap = 'jet'
            reverseMap = '0'
            cOut = ' -1 -1 -1'
            clim = '-0.01 0.01'
            posiFile =  os.path.join(_wdir, 'fields\Step_%d_Cam_%d_imgPts_cellSize_%s.csv' % (istep+1,icam+1,csize))
            fieldFile = os.path.join(_wdir, 'fields\Step_%d_Cam_%d_%s_cellSize_%d.csv' % (istep+1,icam+1,field,csize))
            clfilestr = '010'
            saveImgFile = os.path.join(_wdir, 'fields\Cam_%d_%s_cellSize_%d_cl%s' % (icam+1,field,csize,clfilestr),
                                       'Step_%d_Cam_%d_%s_cellSize_%d_cl%s.JPG' % (istep+1,icam+1,field,csize,clfilestr))
            drawField(_bgImgFile=bgImgFile,
                      _fieldFile=fieldFile,
                      _posiFile=posiFile,
                      _saveImgFile=saveImgFile,
                      _colormap=colormap,
                      _reverseMap=reverseMap,
                      _cOut=cOut,
                      _clim=clim,
                      _viewWidth=viewWidth)
            toc = time.time()
            print("Spc %d Step %d view fields completed. (in %f sec.)"\
                  % (ispc+1, istep+1, toc-tic))


def icf_drawFields(_wdir=None, 
                    _steps=None,
                    _cams=None,
                    _cellSizes=None,
                    _fields=None,
                    _cmaps=None,
                    _reverseMaps=None,
                    _cOutliers=None,
                    _clims=None,
                    _viewWidths=None,
                    _outDirSubstrs=None,
                    ):
    # wdir
    if type(_wdir) == type(None):
        print('# Enter working directory:')
        print("#   E.g., D:\yuansen\ImPro\improMeasure\examples\2022rcwall")
        _wdir = input2()
    wdir = _wdir
    
    # steps 
    if type(_steps) == type(None):
        print("# Enter range of steps:")
        print("#   E.g., 1 300  (for steps from 1 to 300)")
        print("#   E.g., 100 100 (for only step 100)")
        _steps = input2()
    #   convert ' 1 300 ' ==> range(0,300) for loop of cameras
    #   range(0,300) in python is like, [0,1,2,...,299]
    _steps_range = _steps.split()
    steps = range(int(_steps_range[0]) - 1, int(_steps_range[1]))

    # cams
    if type(_cams) == type(None):
        print("# Enter camera IDs:")
        print("#   E.g., 1 2  (for both cameras 1 and 2)")
        print("#   E.g., 1    (for only camera 1)")
        _cams = input2()
    #   convert ' 1 2 ' ==> [0,1] for loop of cameras  
    _cams_list = _cams.split()
    cams = []
    for s in _cams_list:
        cams.append(int(s) - 1)

    # cellSizes
    if type(_cellSizes) == type(None):
        print("# Enter cell sizes one by one:")
        print("#   E.g., 15 60 60 60 60 60 60")
        print("#   Note: The expression of cellSizes is different from that of icf_wallMonitor. In icf_wallMonitor_vx if you define cellSizes to 15 30 60 then they apply to all fields because every cell size needs to run ux and uy, then all derived fields are calculated with little extra cost.")
        print("#         But here drawing pictures for every field of every cell size is at high computing cost. You would like to assign them one by one.")
        print("#         You need to be consistent with number of inputs since now.")
        print("#         For example, If you assign 7 numbers for cell sizes, it means you want to plot 7 fields, so you will input 7 numbers or fields for later inputs such as fields, colormaps, ...")
        _cellSizes = input2()
    _cellSizes_list = _cellSizes.split()
    cellSizes = []
    for s in _cellSizes_list:
        cellSizes.append(int(s))

    # fields
    if type(_fields) == type(None):
        print("# Enter fields:")
        print("#   crack: crack opening field")
        print("#      ux: displacement field along x")
        print("#      uy: displacement field along y")
        print("#     exx: strain field x")
        print("#     eyy: strain field y")
        print("#     gxy: strain field of shear gamma xy")
        print("#      e1: principal strain 1")
        print("#      e2: principal strain 2")
        print("#     gmx: strain field of maximum shear")
        print("#     th1: angle of e1")
        print("#     th2: angle of e2")
        print("#     thg: angle of gmx")
        print("#   E.g., crack ux uy gxy e1 e2 gmx")
        _fields = input2()
    fields = _fields.split()
    
    # colormaps (cmaps)
    if type(_cmaps) == type(None):
        print("# Enter colormaps one by one:")
        print("#   Popular colormaps: bone, jet, viridis")
        print("#   E.g., bone jet jet jet jet jet jet")
        _cmaps = input2()
    cmaps = _cmaps.split()
    
    # reverse colormap or not (reverseMaps)
    if type(_reverseMaps) == type(None):
        print("# Do you want to reverse the colormap (0: No. 1: Yes)?: ")
        print("#   E.g., 1  0 0 0  0 0 0 ")
        _reverseMaps = input2()
    reverseMaps = _reverseMaps.split()
    
    # color of outliers (cOutliers)
    if type(_cOutliers) == type(None):
        print("# Enter color of outliers one by one. Each color contains three integers. ")
        print("#   If color is -1 -1 -1, it means the outliers are plotted based on colormap.")
        print("    Add a comma between colors of different fields.")
        print("#   E.g., 255 255 255, -1 -1 -1, -1 -1 -1, -1 -1 -1, -1 -1 -1, -1 -1 -1, -1 -1 -1")
        _cOutliers = input2()
    cOutliers = _cOutliers.split(sep=',')
    
    # limits of colormap (clims)
    if type(_clims) == type(None):
        print("# Enter the lower and upper bounds of each field one by one:")
        print("    Add a comma between bounds of different fields.")
        print("#   E.g., 0 0.1, -0.01 0.01, -0.01 0.01, -0.01 0.01, -0.01 0.01, -0.01 0.01, -0.01 0.01")
        _clims = input2()
    clims = _clims.split(sep=',')
        
    # image widths (viewWidths)
    if type(_viewWidths) == type(None):
        print("# Enter width of output images:")
        print("#   E.g., 400 400 400 400 400 400 400")
        _viewWidths = input2()
    viewWidths = _viewWidths.split()
    
    # substrings of output directories and files (outDirSubstrs)
    if type(_outDirSubstrs) == type(None):
        print("# Enter sub-strings of output directories and files:")
        print("#   The output image files (.JPG) will be placed at [working dir]/fields/Cam_[camId]_[field]_cellSize_[cellSize]_[sub_string].")
        print("#   For example, if you enter clim20, the output file of ux could be under directory of field/Cam_1_ux_cellSize_15_clim20")
        print("#   E.g., cl01 cl001 cl001 cl001 cl001 cl001 cl001")
        _outDirSubstrs = input2()
    outDirSubstrs = _outDirSubstrs.split()
    #
    nFields = len(fields)
    if len(cellSizes) != nFields:
        print("# ERROR: icf_drawFields(): # of cell size is %d but should be the same as # of fields (%d)" % (len(cellSizes), nFields))
        return 
    if len(cmaps) != nFields:
        print("# ERROR: icf_drawFields(): # of color maps is %d but should be the same as # of fields (%d)" % (len(cmaps), nFields))
        return 
    if len(reverseMaps) != nFields:
        print("# ERROR: icf_drawFields(): # of reverse colormap is %d but should be the same as # of fields (%d)" % (len(reverseMaps), nFields))
        return 
    if len(cOutliers) != nFields:
        print("# ERROR: icf_drawFields(): # of outlier colors is %d but should be the same as # of fields (%d)" % (len(cOutliers), nFields))
        return 
    if len(clims) != nFields:
        print("# ERROR: icf_drawFields(): # of lower/upper bounds is %d but should be the same as # of fields (%d)" % (len(clims), nFields))
        return 
    if len(viewWidths) != nFields:
        print("# ERROR: icf_drawFields(): # of view widths is %d but should be the same as # of fields (%d)" % (len(viewWidths), nFields))
        return 
    if len(outDirSubstrs) != nFields:
        print("# ERROR: icf_drawFields(): # of output directory sub-strings is %d but should be the same as # of fields (%d)" % (len(outDirSubstrs), nFields))
        return 
    #
    for istep in steps:
        for icam in cams:
            for iField in range(nFields):
                field = fields[iField]
                csize = cellSizes[iField]
                colormap = cmaps[iField]
                reverseMap = reverseMaps[iField]
                cOut = cOutliers[iField]
                clim = clims[iField]
                viewWidth = viewWidths[iField]
                posiFile =  os.path.join(_wdir, 'fields/Step_%d_Cam_%d_imgPts_cellSize_%s.csv' % (istep+1,icam+1,csize))
                fieldFile = os.path.join(_wdir, 'fields/Step_%d_Cam_%d_%s_cellSize_%d.csv' % (istep+1,icam+1,field,csize))
                clfilestr = outDirSubstrs[iField]
                saveImgFile = os.path.join(_wdir, 'fields\Cam_%d_%s_cellSize_%d_cl%s' % (icam+1,field,csize,clfilestr),
                                           'Step_%d_Cam_%d_%s_cellSize_%d_cl%s.JPG' % (istep+1,icam+1,field,csize,clfilestr))
                bgImgFile = os.path.join(wdir, 'rectf/Step_%d_Cam_%d_Rectf.tiff' % (istep+1, icam+1))
                drawField(_bgImgFile=bgImgFile,
                          _fieldFile=fieldFile,
                          _posiFile=posiFile,
                          _saveImgFile=saveImgFile,
                          _colormap=colormap,
                          _reverseMap=reverseMap,
                          _cOut=cOut,
                          _clim=clim,
                          _viewWidth=viewWidth)

def test1():
    _wdir = r'D:\ExpDataSamples\20220200_NcreeNorth_RcWall\Analysis_improMeasure_Spc_1_Cams1_2'
    _steps = '1 300'
    _cams = '1 2'
    _cellSizes = '15 60 60 60 60 60'
    _fields = 'crack ux gxy e1 e2 gmx'
    _cmaps = 'bone jet jet jet jet jet '
    _reverseMaps = '1 0 0 0 0 0 '
    _cOutliers = '255 255 255, -1 -1 -1, -1 -1 -1, -1 -1 -1, -1 -1 -1, -1 -1 -1'
    _clims = '0 1, -10 10, -0.01 0.01, -0.01 0.01, -0.01 0.01, -0.01 0.01'
    _viewWidths = '400 400 400 400 400 400'
    _outDirSubstrs = 'cl10 cl10 cl010 cl010 cl010 cl010'
    
    icf_drawFields(_wdir, _steps, _cams, _cellSizes, _fields, _cmaps,
                    _reverseMaps, _cOutliers, _clims, _viewWidths, _outDirSubstrs)

    
if __name__ == '__main__':
    test1()
    
    