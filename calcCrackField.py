# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 13:51:32 2023

@author: yuans
"""

import numpy as np
import numba as nb
from math import cos, sin
from numba import njit

@njit
def calcCrackField(field_ux, field_uy):
    _nPtsY = field_ux.shape[0]
    _nPtsX = field_ux.shape[1]
    # generate expanded disp field (expanded by 1 cell at four sides)
    # for finite difference or crack field calculation
    _field_ux_exp = np.zeros((_nPtsY + 2, _nPtsX + 2), dtype=np.float32)
    _field_ux_exp[1:-1,1:-1] = field_ux
    _field_ux_exp[ 0,1:-1] = 2 * field_ux[ 0,:] - field_ux[ 1,:]
    _field_ux_exp[-1,1:-1] = 2 * field_ux[-1,:] - field_ux[-2,:]
    _field_ux_exp[1:-1, 0] = 2 * field_ux[:, 0] - field_ux[:, 1]
    _field_ux_exp[1:-1,-1] = 2 * field_ux[:,-1] - field_ux[:,-2]
    _field_ux_exp[ 0, 0] = 2 * field_ux[ 0, 0] - field_ux[ 1, 1]
    _field_ux_exp[ 0,-1] = 2 * field_ux[ 0,-1] - field_ux[ 1,-2]
    _field_ux_exp[-1, 0] = 2 * field_ux[-1, 0] - field_ux[-2, 1]
    _field_ux_exp[-1,-1] = 2 * field_ux[-1,-1] - field_ux[-2,-2]
    _field_uy_exp = np.zeros((_nPtsY + 2, _nPtsX + 2), dtype=np.float32)
    _field_uy_exp[1:-1,1:-1] = field_uy
    _field_uy_exp[ 0,1:-1] = 2 * field_uy[ 0,:] - field_uy[ 1,:]
    _field_uy_exp[-1,1:-1] = 2 * field_uy[-1,:] - field_uy[-2,:]
    _field_uy_exp[1:-1, 0] = 2 * field_uy[:, 0] - field_uy[:, 1]
    _field_uy_exp[1:-1,-1] = 2 * field_uy[:,-1] - field_uy[:,-2]
    _field_uy_exp[ 0, 0] = 2 * field_uy[ 0, 0] - field_uy[ 1, 1] + 1e6
    _field_uy_exp[ 0,-1] = 2 * field_uy[ 0,-1] - field_uy[ 1,-2] + 1e6
    _field_uy_exp[-1, 0] = 2 * field_uy[-1, 0] - field_uy[-2, 1] + 1e6
    _field_uy_exp[-1,-1] = 2 * field_uy[-1,-1] - field_uy[-2,-2] + 1e6
    # calculate crack field    
    _field_crack = np.zeros((_nPtsY, _nPtsX), dtype=np.float32)
    for i in range(_nPtsY):
        for j in range(_nPtsX):
            maxCrack = np.float32(-1e30)
            ii = i + 1
            jj = j + 1
            uU = np.array([_field_ux_exp[ii-1, jj], _field_uy_exp[ii-1, jj]], dtype=np.float32)
            uD = np.array([_field_ux_exp[ii+1, jj], _field_uy_exp[ii+1, jj]], dtype=np.float32)
            uL = np.array([_field_ux_exp[ii, jj-1], _field_uy_exp[ii, jj-1]], dtype=np.float32)
            uR = np.array([_field_ux_exp[ii, jj+1], _field_uy_exp[ii, jj+1]], dtype=np.float32)
            for th in [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165]:
                costh = cos(th * np.pi / 180.)
                sinth = sin(th * np.pi / 180.)
                if th < 90:
                    uA = (uU * abs(costh) + uL * abs(sinth)) / (abs(costh) + abs(sinth))
                    uB = (uD * abs(costh) + uR * abs(sinth)) / (abs(costh) + abs(sinth))
                else:
                    uA = (uD * abs(costh) + uL * abs(sinth)) / (abs(costh) + abs(sinth))
                    uB = (uU * abs(costh) + uR * abs(sinth)) / (abs(costh) + abs(sinth))
                field_crackOpening =  -sinth * (uA[0] - uB[0]) - costh * (uA[1] - uB[1])
                if field_crackOpening > maxCrack:
                    maxCrack = field_crackOpening
                    _field_crack[i,j] = field_crackOpening
    return _field_crack

