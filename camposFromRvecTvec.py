import numpy as np
import cv2 as cv

def camposFromRvecTvec(rvec, tvec):
    rvec = np.array(rvec, dtype=float).flatten()
    tvec = np.array(tvec, dtype=float).flatten()
    if rvec.size == 3:
        rmat = cv.Rodrigues(rvec)[0]
    elif rvec.size == 9:
        rmat = cv.Rodrigues(rvec.reshape((3, 3)))[0]
    r44 = np.eye(4, dtype=float)
    r44[0:3, 0:3] = rmat.copy()
    r44[0:3, 3] = tvec[:]
    r44inv = np.linalg.inv(r44)
    return r44inv[0:3, 3].reshape((3, 1))
