import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import glob 
import os
import time
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

inputDir = input("# Enter the input directory: ")
outputDir = input("# Enter the output directory: ")
psize = int(input("# Enter size of patch (e.g., 64): "))
model = tf.keras.models.load_model(input("# Enter model file: "))

previousFiles = []
while True:
    try:
        # step 1: read current files
        currentFiles = glob.glob(os.path.join(inputDir, "*.JPG"))
        # step 2: find new files
        newFiles = set(currentFiles) - set(previousFiles)
        newFiles = list(newFiles)
        # step 3: 
        for i in range(len(newFiles)):
            file = newFiles[i]
            # do something about the file
            print("# Reading %s" % file)
            img = cv.imread(file)
            imgHeight = img.shape[0]
            imgWidth = img.shape[1]
            img = img.reshape((imgHeight, imgWidth, -1))
            # calculate border
            x0 = (imgWidth % psize) // 2
            y0 = (imgHeight % psize) // 2
            npY = imgHeight // psize 
            npX = imgWidth // psize
            # create a 2D array for probability and newImg
            prob = np.zeros((npY, npX), dtype=float) 
            newImg = img.copy()
            for iY in range(npY):
                for iX in range(npX):
                    # each patch 
                    yStart = y0 + iY * psize
                    yEnd = yStart + psize 
                    xStart = x0 + iX * psize
                    xEnd = xStart + psize
                    imgPatch = img[yStart:yEnd, xStart:xEnd, :].copy()
                    ## run analysis and get probability of crack 
#                    pthreshold = 0.02
                    p = model.predict(imgPatch.reshape(1,psize,psize,-1))[0,1]
#                    print("Patch %d %d - p=%f" % (iY, iX, p))
#                    p = p / pthreshold
#                    p = max(0, min(1., p))
#                    cv.imshow("Patch %d %d" % (iY, iX), imgPatch); cv.waitKey(0); cv.destroyAllWindows()
                    ## draw on the new image: 
                    option = 2
                    if option == 1:
                        if (p > 0.5): 
                            start_point = (xStart, yStart)
                            end_point = (xEnd - 2, yEnd - 2)
                            cv.rectangle(newImg, start_point, end_point, (0, 255, 0), 2)
                    elif option == 2:
                        if (p <= 0.5):                            
                            patch = newImg[yStart:yEnd, xStart:xEnd, :].copy()
                            newImg[yStart:yEnd, xStart:xEnd, :] += \
                                (np.ones(patch.shape, dtype=np.uint8)*255 - patch) // 2
            outputFile = os.path.join(outputDir, os.path.split(file)[1])
            cv.imwrite(outputFile, newImg)

        # step 4: update
        previousFiles = currentFiles
        time.sleep(1.0)            
    except: 
        break

"""
Test1:
D:/yuansen/ImPro/improMeasure/examples/crackTraining/test_monitoring/Input
D:/yuansen/ImPro/improMeasure/examples/crackTraining/test_monitoring/Output
64
D:/yuansen/ImPro/improMeasure/examples/crackTraining/model_patch_64_64_c323232_d32.h5

"""
