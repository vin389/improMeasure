import os
import time
import numpy as np
import cv2 as cv
import glob
from numba import njit

# This computing is a part of metric rectification.
# It generates a grid of dense 3D points given vectors x and z.
h,w = 2522, 1671
vxp = np.array([ 1.05096145, -0.01773866, -0.00130174], dtype=float)
vzp = np.array([ 0.00125499, -0.00278092,  1.05110752], dtype=float)

# The natural version: 9.36 sec. (h,w=2522,1671)
tic = time.time()
g = np.zeros((h * w, 3), dtype=float)
for i in range(h):
    for j in range(w):
        g[i * w + j] = j * vecx - i * vecz
toc = time.time()
print(toc-tic)
_g = g.copy()

# Simple manual optimization: 5.888 sec.
tic = time.time()
g = np.zeros((h * w, 3), dtype=float)
cij = 0
for i in range(h):
    _ivecz = i * vecz
    for j in range(w):
        g[cij] = j * vecx - _ivecz
        cij += 1
toc = time.time()
print(toc-tic)
print(np.max(np.abs(g.flatten() - _g.flatten())))

# Numba njit for the natural version: 1st: 0.469 sec., nth: 0.347 sec.
tic = time.time()
@njit
def g_jit(h, w, vecx, vecz, g):
    cij = 0
    for i in range(h):
        _ivecz = i * vecz
        for j in range(w):
            g[cij] = j * vecx - _ivecz
            cij += 1
toc = time.time()
print(toc-tic)
tic = time.time()
g = np.zeros((h_rectf * w_rectf, 3), dtype=float)
g_jit(h, w, vecx, vecz, g)
toc = time.time()
print(toc-tic)
print(np.max(np.abs(g.flatten() - _g.flatten())))

# Re-form formula (to matrix form): 0.044 sec.
tic = time.time()
mij = np.zeros((2, h * w), dtype=float)
mij0, mij1 = np.meshgrid(range(w), range(h))
mij[0] = mij0.flatten()
mij[1] = mij1.flatten()
mvec = np.array([vecx,-vecz]).transpose()
mgx = np.matmul(mvec, mij) 
g = mgx.transpose()
toc = time.time()
print(toc-tic)
print(np.max(np.abs(g.flatten() - _g.flatten())))
