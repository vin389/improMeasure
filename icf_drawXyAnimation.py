import os
import matplotlib.pyplot as plt
import numpy as np

from inputs import input2

def icf_drawXyAnimation(_fileData=None,
                        _filePhotoSteps=None,
                        _outDir=None,
                        _xlabel=None,
                        _ylabel=None,
                        _title=None,
                        ):
    if type(_fileData) == type(None):
        print("# Enter the file of complete X-Y data:")
        print("#   The file should be in CSV format. The first column is the X, and the second column is the Y data.")
        print(r"#   E.g., D:\yuansen\ImPro\improMeasure\examples\xyanim\daqData.csv")
        _fileData = input2()
    fileData = _fileData
    data = np.loadtxt(fileData, dtype=np.float32, delimiter=',')
    print("# Your data shape is %d by %d." % (data.shape[0], data.shape[1]))

    if type(_filePhotoSteps) == type(None):
        print("# Do you want do draw all records of the data?")
        print("#   If yes, enter a single dot and ENTER.")
        print("#   Otherwise, enter the file that describes how you would select Xs to plot the data.")
        print("#     The file should be in CSV format containing two columns.")
        print("#     The first column is the image ID you want to output, and the second column is the X this photo maps to.")
        print("#     For example, the following file means 1st image is data X=1; 8th image is data X=100; the 2nd to 7th images will be interpolated.")
        print("#        1, 1")
        print("#        8, 100")
        print("#      200, 1200")
        print("#      300, 2000")
        print("#     The first row must start from image 1. The last row must end at the last image to output.")
        print(r"#   For example, D:\yuansen\ImPro\improMeasure\examples\xyanim\dataStepsOfPhotos.csv")
        _filePhotoSteps = input2()
    if len(_filePhotoSteps) <= 2:
        _filePhotoSteps = ""
        nData = data.shape[0]
        photoSteps = np.array([1, 1, nData, nData], dtype=np.int32).reshape(2,2)
    else:
        filePhotoSteps = _filePhotoSteps
        photoSteps = np.loadtxt(filePhotoSteps, dtype=np.int32, delimiter=',')
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
#            print('x0: %d' % x0)
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

                        # _outDir=None,
                        # _xlabel=None,
                        # _ylabel=None,
                        # _title=None,
    # directory of output files (outDir)
    if type(_outDir) == type(None):
        print("# Enter the directory of the output files:")
        print(r"#   For example, D:\yuansen\ImPro\improMeasure\examples\xyanim\xyplots")
        _outDir = input2()
    outDir = _outDir
    
    # xlabel
    if type(_xlabel) == type(None):
        print("# Enter the xlabel:")
        print("#   For example, Drift (percentage) ")
        _xlabel = input2()
    xlabel = _xlabel
    
    # ylabel
    if type(_ylabel) == type(None):
        print("# Enter the ylabel:")
        print("#   For example, Shear force (kN) ")
        _ylabel = input2()
    ylabel = _ylabel    

    # title
    if type(_title) == type(None):
        print("# Enter the title:")
        print("#   For example, Hysterestic Loop")
        _title = input2()
    title = _title
    
    # plot in files
    for i in range(len(dataStep)):
        istep = round(dataStep[i]) - 1
        fig, ax = plt.subplots()
        ax.yaxis.grid(True, which='major', linestyle='--') # major grid lines
        ax.xaxis.grid(True, which='major', linestyle='--') # major grid lines
        ax.title.set_text(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        ax.plot(data[:,0], data[:,1], linewidth=1.0, color='#C8C8C8', linestyle='--')
        ax.plot(data[:istep,0], data[:istep,1], linewidth=2.0, color='#000000')
        ax.scatter(data[istep,0], data[istep,1], linewidth=0.0, color='#FF0000', s = 50)
        if os.path.exists(outDir) == False:
            os.makedirs(outDir)
        saveFile = os.path.join(outDir, 'daqData_%04d.JPG' % (i+1))
        plt.savefig(saveFile)
        plt.close(fig)
        if i % 10 == 0:
            print("%d" % i)

def test1():
    icf_drawXyAnimation(
        r"D:\yuansen\ImPro\improMeasure\examples\xyanim\daqData.csv",
        r"D:\yuansen\ImPro\improMeasure\examples\xyanim\dataStepsOfPhotos.csv",
        r"D:\yuansen\ImPro\improMeasure\examples\xyanim\xyplots",
        "Drift (percentage)",
        "Shear force (kN)",
        "Hysterestic Loop",
        )
    
def test2():
    icf_drawXyAnimation(
        r"D:\yuansen\ImPro\improMeasure\examples\xyanim\daqData_short.csv",
        ".", 
        r"D:\yuansen\ImPro\improMeasure\examples\xyanim\xyplots",
        "Drift (percentage)",
        "Shear force (kN)",
        "Hysterestic Loop",
        )
    

if __name__ == '__main__':
    test2()
    
