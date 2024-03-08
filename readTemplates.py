import numpy as np
import re
from inputs import input2

def readTemplates(filename=""):
    pts = np.array([])
    # if filename is given
    if (len(filename) >= 1):
        try:
            pts = np.loadtxt(filename, delimiter=',')
            return pts
        except Exception as e:
            print("# Error: readTemplates() cannot read points from file (%s)."
                   % (filename))
            print("# Exception is", e)
    # ask user if filename is empty
    while (pts.size == 0):
        print("\n# How do you want to input templates:")
        print("#  file: Enter file name of csv format points.")
        print("#   # comments")
        print("#     x1,y1,x01,y01,w1,h1")
        print("#     x2,y2,x02,y02,w2,h2")
        print("#     ...")
        print("#            For example: ")
        print("#            file")
        print("#            .\\examples\\pickTemplates\\pickedTemplates_IMG_0001.csv")
        print("#  manual n: Manually type-in n templates:")
        print("#              For example: ")
        print("#                 manual 3")
        print("#     x1,y1,x01,y01,w1,h1")
        print("#     x2,y2,x02,y02,w2,h2")
        print("#     x3,y3,x03,y03,w3,h3")
        uInput = input2().strip()
        if (uInput == "file"):
            try:
                filename = input2()
                pts = np.loadtxt(filename, delimiter=',')
            except Exception as e:
                print("# Error: readTemplates() cannot read points from file (%s)."
                       % (filename))
                print("# Exception is", e)
                print("# Try again.")
                continue
        if (uInput[0:6] == 'manual'):
            nPoints = int(re.split(',| ', uInput.strip())[1])
            pts = np.ones((nPoints,6), dtype=float) * np.nan
            for i in range(nPoints):
                datInput = input2("").strip()
                datInput = re.split(',| ', datInput)
                pts[i,0] = float(datInput[0])
                pts[i,1] = float(datInput[1])
                pts[i,2] = int(float(datInput[2]))
                pts[i,3] = int(float(datInput[3]))
                pts[i,4] = int(float(datInput[4]))
                pts[i,5] = int(float(datInput[5]))
        if (pts.size > 0):
            print("# Read %d templates" % (pts.shape[0]))
            print("# The first template is ", pts[0])
            print("# The last template is ", pts[-1])
    return pts

