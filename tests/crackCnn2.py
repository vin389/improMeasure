import os, sys, time
import glob
import cv2 as cv
import numpy as np
import pandas as pd
import tkinter as tk
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, models, layers, utils, activations
from tensorflow.keras import losses, optimizers, metrics
from keras.utils.vis_utils import plot_model


def inputdlg(prompt: list, name: str='Input Dialog', numlines: int =1, defaultanswer=[]):
    """
    ThiAs function mimics matlab function inputdlg but only supports limited 
    functionality. Only arguments prompt, name, and defaultanswer are supported.
    The arguments numlines and other arguments of matlab inputdlg are
    not supported yet. 

    Example:
        prompt=['Enter the matrix size for x^2:','Enter the colormap name:'];
        name='Input for Peaks function';
        numlines = 1;
        defaultanswer = ['20', 'hsv']
        answers = inputdlg(prompt, name, numlines, defaultanswer)
        if len(answers) > 0:
            print("The inputs are:")
            print(answers)
        else:
            print("Input dialog is cancelled.")
    """
    answers = []
    edits = []
    window = tk.Tk()
    window.title(name)
    nrow = len(prompt)
    for i in range(nrow):
        frame = tk.Frame(master=window, relief=tk.RAISED, borderwidth = 0)
        frame.grid(row=i, column=0)
        label = tk.Label(master=frame, text=prompt[i])
        label.pack()
        #
        frame = tk.Frame(master=window, relief=tk.RAISED, borderwidth = 1)
        frame.grid(row=i, column=1)
        edits.append(tk.Entry(master=frame, width=100))
        if i < len(defaultanswer): 
            edits[i].insert(0, defaultanswer[i])
        edits[i].pack()
    frame = tk.Frame(master=window, relief=tk.RAISED, borderwidth = 1)
    frame.grid(row=nrow, column=0)
    bt1 = tk.Button(master=frame, text='OK')
    bt1.pack()
    frame = tk.Frame(master=window, relief=tk.RAISED, borderwidth = 1)
    frame.grid(row=nrow, column=1)
    bt2 = tk.Button(master=frame, text='Cancel')
    bt2.pack()
    def eventOK(e):
        for i in range(nrow):
            answers.append(edits[i].get())
        window.destroy()
    def eventCancel(e):
        answers = []
        window.destroy()
    bt1.bind('<Button>', eventOK)
    bt2.bind('<Button>', eventCancel)
    window.mainloop()
    return answers


def trainImgFilesToImgArray(files: list):
    """
    This function converts files (list of file names) to single image array.
    
    Example
    -------
    # In this case, imgFiles would be ['xxx/CFD_001.JPG', 'xxx/CFD_002.JPG', 'xxx/CFD_003.JPG']
    files = glob.glob(r'D:\yuansen\ImPro\improMeasure\examples\crackTraining\images\*.JPG')[0:3]
    # In this case, imgArray would be an np.ndarray (shape=(3,448,448,3), dtype=np.uint8)
    imgArray = trainImgFilesToImgArray(files)
    """
    img = cv.imread(files[0])
    if type(img) == type(None):
        print("# Error: trainImgFilesToImgArray(): Cannot read image from %s" % files[0])
        return
    # get image dimension (imgH, imgW, imgNc) (even if it has only one channel)
    img = img.reshape(img.shape[0], img.shape[1], -1)
    (imgH, imgW, imgNc) = img.shape
    # allocate image array
    imgArray = np.zeros((len(files), imgH, imgW, imgNc), dtype=np.uint8)
    # read image and fill in imgArray
    for i in range(len(files)):
        img = cv.imread(files[i])
        if type(img) == type(None):
            print("# Error: trainImgFilesToImgArray(): Cannot read image from %s" % files[i])
            return
        img = img.reshape(img.shape[0], img.shape[1], -1)
        (imgHi, imgWi, imgNci) = img.shape
        if imgHi != imgH or imgWi != imgW or imgNci != imgNc:
            print("# Warning: The %d-th file has inconsistent image size (or number of channels)." % (i + 1))
            continue
        imgArray[i,:,:,:] = img[:,:,:]
    return imgArray        


def imgGridSlice(images: np.ndarray, cellHeight: int, cellWidth: int):
    """
    This function slices images into grids of images. For a 448-by-448-pixel image is to 
    be sliced into 49 64-by-64-pixel images (or 49 cells). 

    Parameters
    ----------
    images : np.ndarray shape=(N, imgHeight, imgWidth, nChannel) or 
             shape=(N, cellHeight, cellWidth) if it is gray image
        images to be sliced
    cellHeight : int
        the height of each cell to be sliced
    cellWidth : int
        the width of each cell to be sliced.
    
    Return
    ------
    slicedImgs : np.ndarray shape=(N * n1 * n2, cellHeight, cellWidth, nChannel)
        where n1 = imgHeight // cellHeight, n2 = imgWidth // cellWidth
    
    Example
    -------
    files = glob.glob(r'D:\yuansen\ImPro\improMeasure\examples\crackTraining\images\*.JPG')[0:3]
    imgArray = trainImgFilesToImgArray(files)
    slicedImgs = imgGridSlice(imgArray, 64, 64)
    """
    if len(images.shape) == 4:
        (N, imgHeight, imgWidth, nChannel) = images.shape
    elif len(images.shape) == 3:
        (N, imgHeight, imgWidth) = images.shape
    else:
        print("# Error: imgGridSlice() requires images to be either 3D or 4D ndarray.")
        return np.array([])
    images4d = images.reshape(N, imgHeight, imgWidth, -1)
    # n1, n2, slicedImgs
    n1 = imgHeight // cellHeight
    n2 = imgWidth // cellWidth
    slicedImgs = np.zeros((N * n1 * n2, cellHeight, cellWidth, nChannel), dtype=np.uint8)
    # loop each image
    idx1 = 0
    y0, x0 = (imgHeight - n1 * cellHeight) // 2, (imgWidth - n2 * cellWidth)
    for iN in range(N):
        for iN1 in range(n1):
            for iN2 in range(n2):
                slicedImgs[idx1, :, :, :] \
                    = images[iN, y0 + iN1 * cellHeight: y0 + (iN1 + 1) * cellHeight, 
                                   x0 + iN2 * cellWidth: x0 + (iN2 + 1) * cellWidth, 0:nChannel].reshape(
                                    1, cellHeight, cellWidth, nChannel)
                # cv.imshow("TEST %d %d" % (iN1, iN2), slicedImgs[idx1, :, :, :])
                # cv.waitKey(0)
                # cv.destroyAllWindows()
                idx1 += 1
    return slicedImgs


def slicedMasksToCategorical(slicedMasks: np.ndarray):
    """
    This function converts sliced to categorical "With" and "Without." 
    
    Example
    -------
    mskFiles = glob.glob(r'D:\yuansen\ImPro\improMeasure\examples\crackTraining\masks\*.JPG')[0:5]
    # mskFiles would be ['.../CFG_001.JPG', '.../CFG_002.JPG', ..., '.../CFG_005.JPG']
    mskArray = trainImgFilesToImgArray(mskFiles)
    # mskArray would be an np.ndarray shaped (5, 448, 448, 3), where the 5 is number of files
    slicedMsks = imgGridSlice(mskArray, 64, 64)
    # slicedMsks would be an np.ndarray shaped (245, 64, 64, 3), where 245 is 5 * (448//64) * (448//64)
    categ = slicedMasksToCategorical(slicedMsks)
    # categ would be an np.array shaped (147, 2) where 147 is the number of sliced images, and the
    # [0] is the probability of being black (zero, or has no, or without) in type np.float32
    # [1] is the probability of being white (255, or has, or with), in type np.float32
    """
    nimg = slicedMasks.shape[0]
    categ = np.zeros((nimg, 2), dtype=np.float32)
    for i in range(nimg):
        countZero = np.sum(slicedMasks[i,:,:,:] == 0)
        countNonz = np.sum(slicedMasks[i,:,:,:] != 0)
        categ[i,0] = countZero / (countZero + countNonz)
        categ[i,1] = countNonz / (countZero + countNonz)
    return categ


def trainImgMskFilesToImgArrayCateg(imgFiles, mskFiles, cellHeight, cellWidth):
    """
    This function converts files of images and masks to categorical training data x/y.

    Example:
    --------
    imgFiles = glob.glob(r'D:\yuansen\ImPro\improMeasure\examples\crackTraining\images\*.JPG')[0:5]
    mskFiles = glob.glob(r'D:\yuansen\ImPro\improMeasure\examples\crackTraining\masks\*.JPG')[0:5]
    imgArray = trainImgFilesToImgArray(imgFiles)
    mskArray = trainImgFilesToImgArray(mskFiles)
    slicedImgs = imgGridSlice(imgArray, 64, 64)
    slicedMsks = imgGridSlice(mskArray, 64, 64)
    categ = slicedMasksToCategorical(slicedMsks)

    """
    imgArray = trainImgFilesToImgArray(imgFiles)
    mskArray = trainImgFilesToImgArray(mskFiles)
    slicedImgs = imgGridSlice(imgArray, cellHeight, cellWidth)
    slicedMsks = imgGridSlice(mskArray, cellHeight, cellWidth)
    categ = slicedMasksToCategorical(slicedMsks)
    return slicedImgs, slicedMsks, categ


def showImgMskCateg(slicedImgs, slicedMsks, categ, showDim):
    """
    Example:
    --------
    imgFiles = glob.glob(r'D:\yuansen\ImPro\improMeasure\examples\crackTraining\images\*.JPG')[0:5]
    mskFiles = glob.glob(r'D:\yuansen\ImPro\improMeasure\examples\crackTraining\masks\*.JPG')[0:5]
    imgArray = trainImgFilesToImgArray(imgFiles)
    mskArray = trainImgFilesToImgArray(mskFiles)
    slicedImgs = imgGridSlice(imgArray, 64, 64)
    slicedMsks = imgGridSlice(mskArray, 64, 64)
    categ = slicedMasksToCategorical(slicedMsks)
    showImgMskCateg(slicedImgs[0:49,:,:,:], slicedMsks[0:49,:,:,:], categ, (7, 7))
    """
#    if showDim[0] > 9 or showDim[1] > 9:
#        print("# Error: showImgMskCateg() So far we only support dimension <= 9")
#        return
    # check
    if slicedImgs.shape[0] != slicedMsks.shape[0]:
        print("# Error: showImgMskCateg(): slicedImgs and slicedMsks must have the same dimension.")
        return
    #
    global ipage, npage
    ipage = 0
    npage = slicedImgs.shape[0] // (showDim[0] * showDim[1])
    figImg = plt.figure()
    figMsk = plt.figure()
    listAx = []
    print("In showImgMskCateg()1 ipage is %d" % ipage)
    def plotImgMsk():
        global ipage, npage
        figImg.clf()
        figMsk.clf()
        print("In plotImgMsk() ipage is %d" % ipage)
        staIdx = ipage * (showDim[0] * showDim[1])
        endIdx = (ipage + 1) * (showDim[0] * showDim[1])
        figImg.suptitle('Images (%d-%d)' % (staIdx, endIdx - 1))
        for i in range(showDim[0]):
            for j in range(showDim[1]):
                # show image
                idx = ipage * (showDim[0] * showDim[1]) + i * showDim[1] + j
                if idx < slicedImgs.shape[0]:
                    ax = figImg.add_subplot(showDim[0],showDim[1], i * showDim[1] + j + 1)
                    ax.imshow(cv.cvtColor(slicedImgs[idx, :, :, :], cv.COLOR_BGR2RGB))
                    ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False)
        figMsk.suptitle('Masks (%d-%d)' % (staIdx, endIdx - 1))
        for i in range(showDim[0]):
            for j in range(showDim[1]):
                # show image
                idx = ipage * (showDim[0] * showDim[1]) + i * showDim[1] + j
                if idx < slicedMsks.shape[0]:
                    ax = figMsk.add_subplot(showDim[0],showDim[1], i * showDim[1] + j + 1)
                    ax.imshow(cv.cvtColor(slicedMsks[idx, :, :, :], cv.COLOR_BGR2RGB))
                    ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False)
                    listAx.append(ax)
        figImg.canvas.draw()
        figMsk.canvas.draw()
    print("In showImgMskCateg()2 ipage is %d" % ipage)
    plotImgMsk()
    def onclick(event):
        global ipage, npage
        print("In onclick()1 ipage is %d" % ipage)
        thisAx = -1
        staIdx = ipage * (showDim[0] * showDim[1])
        endIdx = (ipage + 1) * (showDim[0] * showDim[1])
        for i in range(len(listAx)):
            if event.inaxes == listAx[i]:
                thisAx = i
        if thisAx >= 0:
            figMsk.suptitle('Masks: (%d-%d). Index=%d(1-based); P of black= %.3f ; P of white= %.3f' % 
                (staIdx, endIdx - 1, thisAx + 1 , categ[thisAx,0], categ[thisAx,1]))
            figMsk.canvas.draw()
        else:
            ipage = ipage + 1
            if ipage >= npage:
                ipage = 0
            plotImgMsk()
        print("In onclick()2 ipage is %d" % ipage)
    cid = figMsk.canvas.mpl_connect('button_press_event', onclick)
#    cid = figMsk.canvas.mpl_connect('button_press_event', lambda event: onclick(event, listAx, figMsk, categ, ipage))
    plt.show(block=True)

patchSize = 64
imgFiles = glob.glob(r'D:\yuansen\ImPro\improMeasure\examples\crackTraining\images\*.JPG')
mskFiles = glob.glob(r'D:\yuansen\ImPro\improMeasure\examples\crackTraining\masks\*.JPG')
traindataFile = r'D:\yuansen\ImPro\improMeasure\examples\crackTraining\train_patch_%d_%d_xy.npz' % (patchSize, patchSize)
trainedModelFile = r'D:\yuansen\ImPro\improMeasure\examples\crackTraining\model_patch_%d_%d.h5' % (patchSize, patchSize))


imgArray = trainImgFilesToImgArray(imgFiles)
mskArray = trainImgFilesToImgArray(mskFiles)
slicedImgs = imgGridSlice(imgArray, patchSize, patchSize)
slicedMsks = imgGridSlice(mskArray, patchSize, patchSize)
categ = slicedMasksToCategorical(slicedMsks)
# showImgMskCateg(slicedImgs[0:49,:,:,:], slicedMsks[0:49,:,:,:], categ, (7, 7))

# save training data to npz file
np.savez_compressed(traindataFile, x_train=slicedImgs, y_train=categ)

# load training data from npz file
trFileLoaded = np.load(traindataFile)
x_train_all = trFileLoaded['x_train']
y_train_all = trFileLoaded['y_train']

threshold = 0.01
nbalanced = 0
x_train_bal = x_train_all.copy()
y_train_bal = y_train_all.copy()
# convert to binary classification (no crack: [1 0], has crack: [0 1])
for i in range(y_train_all.shape[0]):
    if y_train_all[i,1] > threshold:
        y_train_all[i,1] = 1
    else:
        y_train_all[i,1] = 0
    y_train_all[i,0] = 1 - y_train_all[i,1]
# balance the classification
rbalance = int(np.sum(y_train_all[:,:]) / np.sum(y_train_all[:,1]) + 0.5)
for i in range(y_train_all.shape[0]):
    if (y_train_all[i,1] == 1 or i % rbalance == 0):
        x_train_bal[nbalanced,:,:,:] = x_train_all[i,:,:,:]
        y_train_bal[nbalanced,:] = y_train_all[i,:]
        nbalanced = nbalanced + 1

x_train_bal = x_train_bal[0:nbalanced,:,:,:].copy()
y_train_bal = y_train_bal[0:nbalanced,:].copy()

useBalanced = True
if useBalanced:
    totalCount = x_train_bal.shape[0]
    validRatio = 0.1
    trainCount = int(totalCount * (1 - 0.1) + 0.5)
    validCount = totalCount - trainCount
    x_train = x_train_bal[0:trainCount,:,:,:].copy()
    x_valid = x_train_bal[trainCount:totalCount,:,:,:].copy()
    y_train = y_train_bal[0:trainCount,:].copy()
    y_valid = y_train_bal[trainCount:totalCount,:].copy()
else:
    totalCount = x_train_all.shape[0]
    validRatio = 0.1
    trainCount = int(totalCount * (1 - 0.1) + 0.5)
    validCount = totalCount - trainCount
    x_train = x_train_all[0:trainCount,:,:,:].copy()
    x_valid = x_train_all[trainCount:totalCount,:,:,:].copy()
    y_train = y_train_all[0:trainCount,:].copy()
    y_valid = y_train_all[trainCount:totalCount,:].copy()

img_size = x_train.shape[1]
num_classes = y_train.shape[1]
nChannel = x_train.shape[3]

if (x_train.shape[2] != x_train.shape[1]):
    print("# Error: Training image has to be a square (height=width)")

ipt = layers.Input(shape=(img_size, img_size, nChannel))
cn1 = layers.Conv2D(filters=32, kernel_size=3, activation='relu')(ipt)
mx1 = layers.MaxPooling2D(pool_size=(2, 2))(cn1)
cn2 = layers.Conv2D(filters=32, kernel_size=3, activation='relu')(mx1)
mx2 = layers.MaxPooling2D(pool_size=(2, 2))(cn2)
cn3 = layers.Conv2D(filters=32, kernel_size=3, activation='relu')(mx2)
mx3 = layers.MaxPooling2D(pool_size=(2, 2))(cn3)
gap = layers.GlobalAveragePooling2D()(mx3)
dn1 = layers.Dense(32, activation='relu')(gap)
drp = layers.Dropout(0.4)(dn1)
dn2 = layers.Dense(num_classes, activation='softmax')(drp)
my_model = models.Model(inputs=ipt, outputs=dn2)
my_model.compile(loss=losses.categorical_crossentropy, \
              optimizer = optimizers.Adam(learning_rate=0.0001),
              metrics = ['accuracy'])
my_epochs = 200
tic = time.time()
my_logs = my_model.fit(x_train, y_train, batch_size = 16, \
          epochs = my_epochs, verbose = 1, \
          validation_data= (x_valid, y_valid))
toc = time.time()

my_model.save(r'D:\yuansen\ImPro\improMeasure\examples\crackTraining\model_patch_64_64_c323232_d32.h5')

#my_model.save(r'D:\yuansen\ImPro\improMeasure\examples\crackTraining\model_patch_64_64_c323232_d32.h5')

# 

#my_model = tf.keras.models.load_model('model_patch_64_64_c321616_d16.h5')

# 

