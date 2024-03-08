import time
import cv2 as cv
import numpy as np 

fname='binFileTest.bin'

f = open(fname, 'wb')
while True:
    uinput = input('Enter 3 integers separated with a comma (,) For example: 1,2,3:')
    mat = np.array(eval('[' + uinput + ']'), dtype=np.uint8)
    if mat[0] == 0:
        break
    data = mat.tobytes()
    f.write(data)
    f.seek(0)
    
f.close()

 
       