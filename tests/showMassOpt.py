import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

imgFile = r'D:\ExpDataSamples\20221100_SteelFrames\20221109\FUJI_L_1967\DSC06874.JPG'
#npzFile = r'D:\yuansen\ImPro\improMeasure\examples\massTracking\ofile1.npz'
imgFile = r'D:\ExpDataSamples\20220200_NcreeNorth_RcWall\20220310_Specimen1_ExpImages\C1\IMG_1833.JPG'
npzFile = r'D:\yuansen\ImPro\improMeasure\examples\massTracking\ofile_2022RC.npz'
showW = 1400
showH = 700
winTitle = "Mass Tracking Display"

# load data 
npzData = np.load(npzFile)
points = npzData['points']
status = npzData['status']
errors = npzData['err']
gridDim = npzData['griddim']
winSize = npzData['winsize']
npzData.close()
showImg = np.zeros((showH, showW, 3), dtype=np.uint8)

# background image 
bgImg = cv.imread(imgFile)
bgScale = min(showH / bgImg.shape[0], 0.5 * showW / bgImg.shape[1])
bgSizeS = (int(bgImg.shape[1] * bgScale + .5), int(bgImg.shape[0] * bgScale + .5))
# resized (smaller) background image to fit the left part of show window
bgImgS1 = cv.resize(bgImg, bgSizeS, interpolation=cv.INTER_LANCZOS4)
# light version of background 
bgImgS2 = bgImgS1 + (np.ones(bgImgS1.shape, dtype=bgImgS1.dtype) * 255 - bgImgS1) // 2
showImg[0:bgImgS1.shape[0], 0:bgImgS1.shape[1]] = bgImgS2

def mouseEvent(event, x, y, flags, param):
    global gridDim, winSize, bgScale, bgImgS1, bgImgS2, showImg
    # calculate grid coordinate according to how grid is defined.
    #    ys = np.linspace(-0.5 + 0.5 * gridDy, -0.5 + bgImgS1.shape[0] - 0.5 * gridDy, gridDim[0])
    #    xs = np.linspace(-0.5 + 0.5 * gridDx, -0.5 + bgImgS1.shape[1] - 0.5 * gridDx, gridDim[1])
    if y >= 0 and y < bgImgS1.shape[0] and x >= 0 and x < bgImgS1.shape[1]:
        gridDy = bgImgS1.shape[0] / gridDim[0]
        gridDx = bgImgS1.shape[1] / gridDim[1]
        gy = (y * gridDim[0]) // bgImgS1.shape[0] 
        gx = (x * gridDim[1]) // bgImgS1.shape[1]
        gy = min(max(gy, 0), gridDim[0] - 1)
        gx = min(max(gx, 0), gridDim[1] - 1)
        # calculate window (scaled to the shown background) 
        winSizeSx = int(winSize[0] * bgScale + .5)
        winSizeSy = int(winSize[1] * bgScale + .5)
        y0 = max(0, min(int((gy + .5) * gridDy - .5 * winSizeSy + .5), bgImgS1.shape[0]))
        y1 = max(0, min(int((gy + .5) * gridDy + .5 * winSizeSy + .5), bgImgS1.shape[0]))
        x0 = max(0, min(int((gx + .5) * gridDx - .5 * winSizeSx + .5), bgImgS1.shape[1]))
        x1 = max(0, min(int((gx + .5) * gridDx + .5 * winSizeSx + .5), bgImgS1.shape[1]))
        # make window the original color while other parts the lightened color 
        showImg[0:bgImgS1.shape[0], 0:bgImgS1.shape[1]] = bgImgS2
        showImg[y0:y1, x0:x1, :] = bgImgS1[y0:y1, x0:x1, :]    
        # get displacement data
        pointIdx = gy * gridDim[1] + gx  
        tStep = np.arange(1, points.shape[0] + 1)
        ux = points[:,pointIdx,0].reshape(-1)
        ux = ux - ux[0]
        # plot data on a canvas (using Matplotlib) 
        dpi=100
        fig = Figure(figsize=(showW/2/dpi, showH/4/dpi), dpi=dpi)
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        ax.plot(tStep, ux); ax.grid(True)
        canvas.draw()
        buf = canvas.buffer_rgba()
        buf = cv.cvtColor(np.asarray(buf), cv.COLOR_RGBA2BGR)
        buf = cv.resize(buf, (showImg.shape[1] // 2, showImg.shape[0] // 4))
        y0 = 0; y1 = y0 + buf.shape[0]
        x0 = showImg.shape[1] // 2; x1 = x0 + buf.shape[1] 
        showImg[y0:y1, x0:x1] = buf
        cv.imshow(winTitle, showImg)
        print("\b" * 100, end='')
        print("Grid coordinate: %4d %4d. Idx: %d " % (gy, gx, pointIdx), end='', flush=True)

cv.namedWindow(winTitle)
cv.setMouseCallback(winTitle, mouseEvent)

cv.imshow(winTitle, showImg)
cv.waitKey(0)
try:
    cv.destroyWindow(winTitle)
except: 
    pass


