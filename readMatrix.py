import numpy as np
import re

from inputs import input2

def readMatrix(filename=''):
    mat = np.array([])
    # if filename is given
    if (len(filename) >= 1):
        try:
            mat = np.loadtxt(filename, delimiter=',')
            return mat
        except Exception as e:
            print("# Error: readMatrix() cannot read matrix from file (%s)."
                   % (filename))
            print("# Exception is", e)
            print("# The file needs to be in csv format.")
     # ask user if filename is empty
    while (mat.size == 0):
        print("\n# How do you want to input templates:")
        print("#     file: reading matrix from file (csv format)")
        print("#        For example: ")
        print("#           file")
        print("#           ..\\..\\examples\\mat.csv")
        print("#     manual m n: reading m-by-n matrix by typing:")
        print("#       for example:")
        print("#         manual 2 3")
        print("#           0.0, 1.1, 2.2")
        print("#           1.2, 2.3, 3.4")
        uInput = input2().strip()
        if (uInput == "file"):
            try:
                filename = input2()
                mat = np.loadtxt(filename, delimiter=',')
            except Exception as e:
                print("# Error: readMatrix() cannot read matrix from file (%s)."
                       % (filename))
                print("# Exception is", e)
                print("# Try again.")
                continue
        if (uInput[0:6] == 'manual'):
            m = int(uInput.split()[1])
            n = int(uInput.split()[2])
            mat = np.ones((m,n), dtype=float) * np.nan
            print("#   Enter all element in this %dx%d matrix:" % (m, n))
            print("#    (one line for each row).")
            for i in range(m):
                datInput = input2("").strip()
                datInput = re.split(',| ', datInput)
                for j in range(n):
                    mat[i,j] = float(datInput[j])
        if (mat.size > 0):
            print("# Read %d-by-%d matrix" % (mat.shape[0], mat.shape[1]))
            print("# The first element is ", mat.flatten()[0])
            print("# The last template is ", mat.flatten()[-1])
            break
    return mat