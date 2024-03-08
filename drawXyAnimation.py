import os
import matplotlib.pyplot as plt
import numpy as np

from inputs import input2

if __name__ == '__main__':
    _wdir = r"D:\ExpDataSamples\20220200_NcreeNorth_RcWall\Analysis_improMeasure_Spc_4_Cams1_2"
    _fileData = r"daqData.csv"
    _filePhotoSteps = r"dataStepsOfPhotos.csv"
    xlabel = "Drift ratio (%)"
    ylabel = "Base shear (kN)"
    title = "Specimen 4 Hysteresis"
    folder = 'daqDataVideo'
    
    fileData = os.path.join(_wdir, _fileData)
    data = np.loadtxt(fileData, dtype=np.float32, delimiter=',')
    
    filePhotoStep = os.path.join(_wdir, _filePhotoSteps)
    photoSteps = np.loadtxt(filePhotoStep, dtype=np.int32, delimiter=',')
    dataStep = -1 * np.ones(np.max(photoSteps[:,0]), dtype=np.int32)
    for i in range(photoSteps.shape[0]):
        x0 = photoSteps[i, 0]
        y0 = photoSteps[i, 1]
        dataStep[x0 - 1] = y0
    
    # interpolate values in dataStep if element is negative
    # dataStep is one-based. First photo is 1, not 0. First step is 1 too.  
    # E.g., dataStep [1 -1 -1 -1 9 -1 -1 -1 17] ==> [1 3 5 7 9 11 13 15 17]
    # dataStep[0] and dataStep[-1] must be positive (cannot be interpolated)
    x0 = -1
    x1 = -1
    for i in range(len(dataStep)):
        # if now we are finding start of interpolation
        if x0 == -1 and dataStep[i] < 0:
            # found start of interpolation x0
            x0 = i
            print('x0: %d' % x0)
            continue
        # if now we are finding end of interpolation 
        if x0 > 0 and dataStep[i] > 0:
            # found end of interpolation x1
            x1 = i
            x0 = x0 - 1
            y0 = dataStep[x0]
            y1 = dataStep[x1]
#            print('x1: %d' % x1)
            # interpolation
            for x in range(x0, x1):
                y = int(y0 + (y1 - y0) * (x - x0) / (x1 - x0))
#                print("%d %f" % (x, y))
                dataStep[x] = y
            x0 = -1
            x1 = -1
    
    # plot in files
    for i in range(len(dataStep)):
        istep = round(dataStep[i])
        fig, ax = plt.subplots()
        ax.yaxis.grid(True, which='major', linestyle='--') # major grid lines
        ax.xaxis.grid(True, which='major', linestyle='--') # major grid lines
        ax.title.set_text(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        ax.plot(data[:,0], data[:,1], linewidth=1.0, color='#C8C8C8', linestyle='--')
        ax.plot(data[:istep,0], data[:istep,1], linewidth=2.0, color='#000000')
        ax.scatter(data[istep,0], data[istep,1], linewidth=0.0, color='#FF0000', s = 50)
        saveFileDir = os.path.join(_wdir, folder)
        if os.path.exists(saveFileDir) == False:
            os.makedirs(saveFileDir)
        saveFile = os.path.join(saveFileDir, 'daqData_%04d.JPG' % (i+1))
        plt.savefig(saveFile)
        plt.close(fig)
#    ax.set_yticks([-200,-100,0,100,200], minor=True) # major ticks
#    ax.yaxis.grid(True, which='major') # major grid lines
    
    
    
    
    
    
    
    