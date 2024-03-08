# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 13:51:32 2023

@author: yuans
"""

import numpy as np
import numba as nb
from math import cos, sin, atan2, pi, sqrt
from numba import njit

@njit
def calcStrainFields(field_ux, field_uy, _dx, _dy):
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
    _field_uy_exp[ 0, 0] = 2 * field_uy[ 0, 0] - field_uy[ 1, 1]
    _field_uy_exp[ 0,-1] = 2 * field_uy[ 0,-1] - field_uy[ 1,-2]
    _field_uy_exp[-1, 0] = 2 * field_uy[-1, 0] - field_uy[-2, 1]
    _field_uy_exp[-1,-1] = 2 * field_uy[-1,-1] - field_uy[-2,-2]
    # calculate strain fields
    # exx is d(ux)/dx, eyy is d(uy)/dy
    # gxy is d(ux)/dy + d(uy)/dx, (not 0.5 d(ux)/dy + 0.5 d(uy)/dx)
    # e1 and e2 are principal strains (2D) 
    # gmx is maximum shear strain
    # th1 and th2 are angles of e1 and e2 respectively (-180 to 180)
    # thg is the angle of maximum shear strain (-180 to 180)
    # angles are closewise from axis x, measured in degrees
    _field_exx = np.zeros((_nPtsY, _nPtsX), dtype=np.float32)
    _field_eyy = np.zeros((_nPtsY, _nPtsX), dtype=np.float32)
    _field_gxy = np.zeros((_nPtsY, _nPtsX), dtype=np.float32)
    _field_e1  = np.zeros((_nPtsY, _nPtsX), dtype=np.float32)
    _field_e2  = np.zeros((_nPtsY, _nPtsX), dtype=np.float32)
    _field_gmx = np.zeros((_nPtsY, _nPtsX), dtype=np.float32)
    _field_th1 = np.zeros((_nPtsY, _nPtsX), dtype=np.float32)
    _field_th2 = np.zeros((_nPtsY, _nPtsX), dtype=np.float32)
    _field_thg = np.zeros((_nPtsY, _nPtsX), dtype=np.float32)
    for i in range(_nPtsY):
        for j in range(_nPtsX):
            ii = i + 1
            jj = j + 1
            exx = (_field_ux_exp[ii,jj+1] - _field_ux_exp[ii,jj-1]) / (2 * _dx)
            eyy = (_field_uy_exp[ii-1,jj] - _field_uy_exp[ii+1,jj]) / (2 * _dy)
            gxy = (_field_ux_exp[ii-1,jj] - _field_ux_exp[ii+1,jj]) / (2 * _dy)\
                + (_field_uy_exp[ii,jj+1] - _field_uy_exp[ii,jj-1]) / (2 * _dx)
            R = sqrt( ((exx - eyy) / 2.) ** 2 + (gxy / 2.) ** 2)
            C = (exx + eyy) / 2.
            e1 = C + R
            e2 = C - R
            gmx = 2 * R
            th1 = 0.5 * atan2(-gxy, exx - eyy) * 180. / pi
            th2 = 0.5 * atan2(gxy, -exx + eyy) * 180. / pi
            thg = (45. - th1) % 360 - 180.
            _field_exx[i,j] = exx
            _field_eyy[i,j] = eyy
            _field_gxy[i,j] = gxy
            _field_e1[i,j]  = e1
            _field_e2[i,j]  = e2
            _field_gmx[i,j] = gmx
            _field_th1[i,j] = th1
            _field_th2[i,j] = th2
            _field_thg[i,j] = thg

    return _field_exx, _field_eyy, _field_gxy, _field_e1, _field_e2,\
           _field_gmx, _field_th1, _field_th2, _field_thg
