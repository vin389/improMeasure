# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 10:47:19 2022

@author: vince
"""

import numpy as np 
import cv2 as cv

def imgToGridImg(img, cellSize):
    """
    If img is 2048 x 2048 picture, and cellSize is
    (32,32), this function returns 64x64 sub-images.
    Each sub-image is a 32-by-32 image. 

    Parameters
    ----------
    img : TYPE
        DESCRIPTION.
    cellSize : TYPE
        DESCRIPTION.

    Returns
    -------
    grid : TYPE
        DESCRIPTION.

    """
    imgWidth = img.shape[1]
    imgHeight = img.shape[0]
    cellWidth = cellSize[0]
    cellHeight = cellSize[1]
   
    
   
    return grid
    
    
    
    
    