
import numpy as np
from numba import njit

@njit
def _test(X, R):
#    B = X    
    X = np.array(X)
    X = X.reshape(4, 1)
    R = R.reshape(4)
    B = X
    return B

if __name__ == '__main__':
    X = np.array([1,2,3,4], dtype=float)
    R = np.array([5,6,7,8], dtype=float)
    print(_test(X, R))
    