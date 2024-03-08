import numpy as np
import cv2 as cv
import time

files=[r'D:\yuansen\ImPro\improMeasure\examples\massTracking\DSC09470.JPG',\
       r'D:\yuansen\ImPro\improMeasure\examples\massTracking\DSC09473.JPG',\
       r'D:\yuansen\ImPro\improMeasure\examples\massTracking\DSC09476.JPG']
gridDim=[90, 120]
winSize=[181, 181]
ofile1 = r'D:\yuansen\ImPro\improMeasure\examples\massTracking\ofile_2022RC'

nfiles = len(files)
npoints = gridDim[0] * gridDim[1]

# allocate memory
points = np.zeros((nfiles, npoints, 2), dtype=np.float32)
status = np.zeros((nfiles, npoints), dtype=np.uint8)
err    = np.zeros((nfiles, npoints), dtype=np.float32)

# define initial points
img0 = cv.imread(files[0], cv.IMREAD_GRAYSCALE)
imgSize = img0.shape
gridDy = imgSize[0] / gridDim[0]
gridDx = imgSize[1] / gridDim[1]
ys = np.linspace(-0.5 + 0.5 * gridDy, -0.5 + imgSize[0] - 0.5 * gridDy, gridDim[0])
xs = np.linspace(-0.5 + 0.5 * gridDx, -0.5 + imgSize[1] - 0.5 * gridDx, gridDim[1])
ipoint = 0
for iy in range(gridDim[0]):    # you may want to use other way to improve the efficiency (griddata?)
    for ix in range(gridDim[1]):
        points[0, ipoint, 0] = xs[ix]
        points[0, ipoint, 1] = ys[iy]
        ipoint += 1

# run loop through files
prevImg = img0.copy()
prevPts = points[0, :, :].reshape(-1, 2)
tic = time.time()
for ifile in range(1, nfiles):
    # timing
    if (ifile > 1):
        toc = time.time()
        nDone = ifile - 1
        nToDo = nfiles - 1 - nDone
        tDone = toc - tic
        tToDo = (tDone / nDone) * nToDo
        print("# Analyzing file %d/%d. Remaining time: %6.1f sec." % (ifile + 1, nfiles - 1, tToDo), end='') 
    # read next image
    nextImg = cv.imread(files[ifile], cv.IMREAD_GRAYSCALE)
    # initial guess of next image points
    nextPts = prevPts.copy()
    # run analysis
    nextPts, st, er = cv.calcOpticalFlowPyrLK(prevImg, nextImg, prevPts, nextPts, winSize=winSize)
    # save data to big array (points, status, err)
    points[ifile, :, :] = nextPts
    status[ifile, :] = st.reshape(-1)
    err[ifile, :] = er.reshape(-1)
    # update next prevPts to current nextPts
    prevPts = nextPts.copy() 
    # timing
    print("\b" * 100, end='', flush=True)
    
print("\n# Done. Total time: %6.1f sec." % (time.time() - tic))

# save files
np.savez_compressed(ofile1, griddim=gridDim, winsize=winSize, points=points, status=status, err=err)
# You can use these statements to load the data
# loaded = np.load(ofile1 + ".npz") 
# print(loaded.files) # --> ['griddim', 'winsize', 'points', 'status', 'err']
# gridDim = loaded['griddim']
# winSize = loaded['winsize']
# points = loaded['points']
# status = loaded['status']
# err = loaded['err']
