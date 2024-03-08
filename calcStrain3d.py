# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 16:50:50 2023

@author: yuans
"""
import numpy as np

from inputs import input2
from q4Bmatrix import q4Bmatrix
from math import sqrt

def principalStrain2D(strains):
    """
    This function calculates the principal strains (e1 and e2) and the maximum
    shear strain (gamma_max) of 2D strain based on the basic Mohr circle 
    calculation.

    Parameters
    ----------
    strains : TYPE numpy.array (N x 3, N is the number of points)
        The 2D strains (exx, eyy, gamma_xy) of N points

    Returns
    -------
    pstrains : TYPE numpy.array (N x 3, N is the number of points)
        The 2D principal strains (e1 and e2, e1 >= e2) and the maximum shear
        (gamma_max)

    """
    n = strains.shape[0]
    pstrains = np.zeros(strains.shape, dtype=strains.dtype)
    for i in range(n):
        exx = strains[i, 0]
        eyy = strains[i, 1]
        gxy = strains[i, 2]
        R = sqrt( ((exx - eyy) / 2.) ** 2 + (gxy / 2.) ** 2)
        C = (exx + eyy) / 2.
        e1 = C + R
        e2 = C - R
        gmx = 2 * R
        # You can uncomment the following lines if you want to get the angles.
        # th1 = 0.5 * atan2(-gxy, exx - eyy) * 180. / pi
        # th2 = 0.5 * atan2(gxy, -exx + eyy) * 180. / pi
        # thg = (45. - th1) % 360 - 180.
        pstrains[i, 0] = e1
        pstrains[i, 1] = e2
        pstrains[i, 2] = gmx
    return pstrains

def calcQ4Strain3d(x0, x1, strains0 = np.zeros((4, 3), dtype=float), principal=False):
    """
    This function calculates 2D strain fields (exx, eyy, gamma_xy) at four gaussian integration points
    of a quadrilateral cell, given the 3D coordinates before and after 
    deformation. 

    Parameters
    ----------
    x0 : TYPE np.array, 4 by 3, float64 or float32
        3D coordinates of 4 points before deformation, in order of counterclock wise.
    x1 : TYPE
        3D coordinates of 4 points after deformation, in order of counterclock wise.
    strains0 : np.array, 4 by 3, float54 or float32, optional
        The strains (principal e1, principal e2, maximum shear strain gamma_xy) at four integration points before deformation.
        The default is np.zeros((4, 3), dtype=np.float32).
        If principal == False, the strain0 is exx, eyy, and gamma_xy
        If principal == True, the strain0 is e1, e2 (e1 >= e2), and gamma_max
    principal : boolean 
        If principal is set to True, the returned values are principal strains 
        (e1 and e2, where e1 >= e2) and the maximum shear strain (gamma_max)
        (rather than exx, eyy, gamma_xy) at four integration points. 
        The dimension remains, 4 by 3. 

    Returns
    -------
    strains1 : TYPE np.array, 4 by 3, float64
        The strains (exx, eyy, gamma_xy) at four integration points after deformation.
        If principal is set to True, the returned strains are principal strains
        e1 and e2, and the maximum shear strain (i.e., e1 - e2). 
        (rather than exx, eyy, gamma_xy)

    """
    x0 = x0.reshape(4, 3).astype('float')
    x1 = x1.reshape(4, 3).astype('float')
    strains1 = np.zeros((4, 3), dtype=float)
    # Find local coordinate system of x0 (--> x0_local)
    x0_local_xvec = (x0[0] - x0[1]) + (x0[3] - x0[2])
    x0_local_yvec = (x0[0] - x0[3]) + (x0[1] - x0[2])
    x0_local_zvec = np.cross(x0_local_xvec, x0_local_yvec)
    x0_local_yvec = np.cross(x0_local_zvec, x0_local_xvec)
    x0_local_xvec = x0_local_xvec / np.linalg.norm(x0_local_xvec)
    x0_local_yvec = x0_local_yvec / np.linalg.norm(x0_local_yvec)
    x0_local_zvec = x0_local_zvec / np.linalg.norm(x0_local_zvec)
    x0_local_orig = .25 * (x0[0] + x0[1] + x0[2] + x0[3])
    m44_x0_local = np.eye(4, dtype=float)
    m44_x0_local[0:3,0] = x0_local_xvec
    m44_x0_local[0:3,1] = x0_local_yvec
    m44_x0_local[0:3,2] = x0_local_zvec
    m44_x0_local[0:3,3] = x0_local_orig
    m44_x0_local_inv = np.linalg.inv(m44_x0_local)
    x0_44 = np.ones((4, 4), dtype=float)
    x0_44[0:3,:] = x0.transpose()
    x0_local = np.matmul(m44_x0_local_inv, x0_44)
    x0_local = x0_local[0:3, :].transpose()
    # Find local coordinate system of x1 (--> x1_local)
    x1_local_xvec = (x1[0] - x1[1]) + (x1[3] - x1[2])
    x1_local_yvec = (x1[0] - x1[3]) + (x1[1] - x1[2])
    x1_local_zvec = np.cross(x1_local_xvec, x1_local_yvec)
    x1_local_yvec = np.cross(x1_local_zvec, x1_local_xvec)
    x1_local_xvec = x1_local_xvec / np.linalg.norm(x1_local_xvec)
    x1_local_yvec = x1_local_yvec / np.linalg.norm(x1_local_yvec)
    x1_local_zvec = x1_local_zvec / np.linalg.norm(x1_local_zvec)
    x1_local_orig = .25 * (x1[0] + x1[1] + x1[2] + x1[3])
    m44_x1_local = np.eye(4, dtype=float)
    m44_x1_local[0:3,0] = x1_local_xvec
    m44_x1_local[0:3,1] = x1_local_yvec
    m44_x1_local[0:3,2] = x1_local_zvec
    m44_x1_local[0:3,3] = x1_local_orig
    m44_x1_local_inv = np.linalg.inv(m44_x1_local)
    x1_44 = np.ones((4, 4), dtype=float)
    x1_44[0:3,:] = x1.transpose()
    x1_local = np.matmul(m44_x1_local_inv, x1_44)
    x1_local = x1_local[0:3, :].transpose()
    # Estimate planar displacement of the membrane (--> u2d)
    u2d = (x1_local - x0_local)[:,0:2].reshape(8)
    # strains_local (4 x 3) = B (3x8) * u2d.flatten() (8)
    strains_local = np.zeros((4, 3), dtype=float)
    gaussian = [-(1./3.)**.5, (1./3.)**.5]
    #   integration point 1     
    r = gaussian[1]; s = gaussian[1]
    B = q4Bmatrix(x0_local[:,0:2], np.array([r, s]))
    strains_local[0, :] = np.matmul(B, u2d)
    #   integration point 2
    r = gaussian[0]; s = gaussian[1]
    B = q4Bmatrix(x0_local[:,0:2], np.array([r, s]))
    strains_local[1, :] = np.matmul(B, u2d)
    #   integration point 3
    r = gaussian[0]; s = gaussian[0]
    B = q4Bmatrix(x0_local[:,0:2], np.array([r, s]))
    strains_local[2, :] = np.matmul(B, u2d)
    #   integration point 4
    r = gaussian[1]; s = gaussian[0]
    B = q4Bmatrix(x0_local[:,0:2], np.array([r, s]))
    strains_local[3, :] = np.matmul(B, u2d)
    strains1 = strains_local
    # Convert to principal strains and maximum shear strain (gamma_max)
    if principal == True:
        strains1 = principalStrain2D(strains1)
    # Plus initial strain
    strains1 += strains0
    return strains1


def icf_calcQ4Strain3d(_x0=None, 
                       _x1=None, 
                       _strains0=None,
                       _principal=None):
    # x0: Coordinates before deformation
    if type(_x0) == type(None):
        print("# Enter 3D coordinates of Q4 nodes (12 reals) BEFORE deformation:")
        print("#   They must be in counter-clockwise order.")
        print("#   E.g., 1 1 0  -1 1 0  -1 -1 0  1 -1 0")
        _x0 = input2()
    x0 = np.fromstring(_x0, sep=' ', dtype=float).reshape(4, 3)
    # x1: Coordinates after deformation 
    if type(_x1) == type(None):
        print("# Enter 3D coordinates of Q4 nodes (12 reals) AFTER deformation:")
        print("#   They must be in counter-clockwise order.")
        print("#   E.g., 1 1 .1  -1 1 -.1  -1 -1 -.1  1 -1 .1")
        _x1 = input2()
    x1 = np.fromstring(_x1, sep=' ', dtype=float).reshape(4, 3)
    # strains0: strains at integration points before deformation
    if type(_strains0) == type(None):
        print("# Enter strains (e1 e2 max_gamma_xy) at four integration points (12 reals) BEFORE deformation:")
        print("#   They must be the same order of Q4 nodes.")
        print("#   E.g., 0 0 0  0 0 0  0 0 0  0 0 0")
        print("#   E.g., 0 (for all zeros)")
        _strains0 = input2()
    if _strains0 == '0':
        strains0 = np.zeros((4, 3), dtype=float)
    else:
        strains0 = np.fromstring(_strains0, sep=' ', dtype=np.float32).reshape(4, 3)
    # principal
    if type(_principal) == type(None):
        print("# Do you want to convert to principal strains and maximum shear strain? (0:False, 1:True)")
        print("#   E.g., 1")
        _principal = input2()
    principal = bool(int(_principal))
        
    # calculation
    strain1 = calcQ4Strain3d(x0, x1, strains0, principal)
    # print
    print("# The strains at integration points (4 by 3) after deformations are:")
    print(strain1)
    return strain1


def test1():
#    _x0 = '1 1 0  -1 1 0  -1 -1 0  1 -1 0'
#    _x1 = '1 1 .1  -1 1 -.1  -1 -1 -.1  1 -1 .1'
    _x0 = '  1  1  0  -1  1   0  -1 -1   0    1 -1   0'
    _x1 = '101 51 .1  99 51  .1  99 49 -.1  101 49 -.1'
    _strains0 = '0'
    _principal = '1'
    strain1 = icf_calcQ4Strain3d(_x0, _x1, _strains0, _principal)

def test2():
    # rigid body motion test (90 degrees along Y)
    _x0 = '  1  1  0  -1  1   0  -1 -1   0    1 -1   0'
    _x1 = '  0  1 -1   0  1   1   0 -1   1    0 -1  -1'
    _strains0 = '0'
    _principal = '1'
    strain1 = icf_calcQ4Strain3d(_x0, _x1, _strains0, _principal)
    print(strain1)
    
def test3():
    # jacobian test (x)
    _x0 = '  10  10  0  -10  10   0  -10 -10   0    10 -10   0'
    _x1 = '  11  10  0  -11  10   0  -11 -10   0    11 -10   0'
    _strains0 = '0'
    _principal = '0'
    strain1 = icf_calcQ4Strain3d(_x0, _x1, _strains0, _principal)
    print(strain1)
    # jacobian test (y)
    _x0 = '  10  10  0  -10  10   0  -10 -10   0    10 -10   0'
    _x1 = '  10  11  0  -10  11   0  -10 -11   0    10 -11   0'
    _strains0 = '0'
    _principal = '0'
    strain1 = icf_calcQ4Strain3d(_x0, _x1, _strains0, _principal)
    print(strain1)
 
def test4():
    # saddle point test
    _x0 = '  1  1  0  -1  1   0  -1 -1   0    1 -1   0'
    _x1 = '  1  1 .1  -1  1 -.1  -1 -1  .1    1 -1 -.1'
    _strains0 = '0'
    _principal = '1'
    strain1 = icf_calcQ4Strain3d(_x0, _x1, _strains0, _principal)
#    print(strain1)
    
if __name__ == '__main__':
    test3()
    
    



