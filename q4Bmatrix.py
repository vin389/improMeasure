# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 16:36:06 2023

@author: yuans
"""

import numpy as np
from numba import njit

# njit cannot be used because Numba does not support matmul??!!
#@njit
def q4Bmatrix(X, R):
    """
    This function returns the B matrix (strain-displacement transformation matrix)
    of a 2D Q4 element, given the 2D coordinates of four nodes (X) and the natural 
    coordinates (R).
    The calculation details can be found in many of finite element textbook, 
    e.g., Bathe, K. J. (1996). Finite Element Procedures, Prentice Hall, USA.

    Parameters
    ----------
    X : TYPE numpy array (4 x 2, float)
        2D coordinates of four nodes, must be in counter-clockwise.
    R : TYPE numpy array (2 x 1 or 1 x 2 or 2, float)
        location of the natural coordinates. R[0] and R[1] must be between 
        -1 and 1. 

    Returns
    -------
    B : TYPE numpy arra (3 x 8, float)
        The B matrix (strain-displacement transformation matrix)
        {strain} = [B] {displacement}
        The {strain} is composed of e_xx, e_yy, and gamma_xy.
        The {displacement} is composed of {ux_1, uy_1, ux_2, ..., uy_4}.
        

    """
#    X = np.array(X).reshape(4, 2)
#    R = np.array(R).reshape(2)
    X = X.reshape(4, 2)
    R = R.reshape(2)
#    J = np.zeros((2, 2), dtype=float)
    J = np.zeros((2,2)).astype(np.float64)
    x1 = X[0, 0]
    y1 = X[0, 1]
    x2 = X[1, 0]
    y2 = X[1, 1]
    x3 = X[2, 0]
    y3 = X[2, 1]
    x4 = X[3, 0]
    y4 = X[3, 1]
    r = R[0]
    s = R[1]
    # Jacobian matrix (J) and its inverse (Jinv)
    J[0, 0] = ((1+s)*x1 - (1+s)*x2 - (1-s)*x3 + (1-s)*x4) / 4.
    J[1, 0] = ((1+r)*x1 + (1-r)*x2 - (1-r)*x3 - (1+r)*x4) / 4.
    J[0, 1] = ((1+s)*y1 - (1+s)*y2 - (1-s)*y3 + (1-s)*y4) / 4.
    J[1, 1] = ((1+r)*y1 + (1-r)*y2 - (1-r)*y3 - (1+r)*y4) / 4.
    Jinv = np.linalg.inv(J)
    # dN/dR, where N is the shape function (Nr, 2 x 4)
    # Nr = 0.25 * np.array([
    #      1+s, -(1+s), -(1-s),  (1-s), 
    #      1+r,   1-r , -(1-r), -(1+r)], dtype=float).reshape(2, 4)
    Nr = 0.25 * np.array([
          1+s, -(1+s), -(1-s),  (1-s), 
          1+r,   1-r , -(1-r), -(1+r)]).astype(np.float64).reshape(2, 4)
    # dN/dX, (2 x 4)
    Nx = np.matmul(Jinv, Nr) 
    # B matrix (3 x 8)    
    B = np.array([
        Nx[0,0],      0., Nx[0,1],      0., Nx[0,2],      0., Nx[0,3],      0, 
              0., Nx[1,0],      0., Nx[1,1],      0., Nx[1,2],      0., Nx[1,3], 
        Nx[1,0], Nx[0,0], Nx[1,1], Nx[0,1], Nx[1,2], Nx[0,2], Nx[1,3], Nx[0,3]],
        dtype=float).reshape(3, 8)
    # B = np.array([
    #     Nx[0,0],       0., Nx[0,1],      0., Nx[0,2],      0., Nx[0,3],      0, 
    #           0., Nx[1,0],      0., Nx[1,1],      0., Nx[1,2],      0., Nx[1,3], 
    #     Nx[1,0], Nx[0,0], Nx[1,1], Nx[0,1], Nx[1,2], Nx[0,2], Nx[1,3], Nx[0,3]]
    #     ).astype(np.float64).reshape(3, 8)
    return B
