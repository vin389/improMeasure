import numpy as np
import scipy.io
import cv2 as cv

from improStrings import npFromStrings

def inputs(prompt=''):
    """
    This function (inputs) is similar to input() but it allows multiple
    lines of input. 
    The key [Enter] does not ends the input. This function reads 
    inputs line by line until a Ctrl-D (or maybe Ctrl-Z), 'end', or 
    'eof' is entered. 
    This function returns a list of strings. 
    
    Parameters
    ----------
    prompt : TYPE, optional
        DESCRIPTION. The default is ''.

    Returns
    -------
    contents : List of strings
        A list of strings. Each string is a line of input.
    """
    print(prompt, end='')
    contents = []
    while True:
        try:
            line = input()
            if line.strip().lower() == 'end' or line.strip().lower() == 'eof':
                raise Exception
        except:
            break
        contents.append(line)
    return contents


def saveImProStereoCalibFileByKeyin():
    print("###############################")
    print("# This function generates a Matlab file (.mat) as the ")
    print("# camera calibration file for ImPro Stereo.")
    print("# This program will ask you to key-in all ")
    print("# parameters (21 to 31) of the left camera,  all ")
    print("# parameters (21 to 31) of the right camera, and the ")
    print("# file path of the .mat file to save.")
    print("# Note: ImPro Stereo only supports distortion coefficients of")
    print("# k1, k2, p1, p2, and k3. Other coefficients (k4, k5, k6, s1 ...)")
    print("# will be ignored and will not be saved to .mat file.")
    print("# Image size can be inputted with any values as they are ignored")
    print("# as well.\n")
    print("# Enter 22 to 31 reals to initialize the Left camera. These reals ")
    print("# are image size (width and height) (2), rvec (3), tvec (3), cmat(9),")
    print("# dvec(4 to 14) and Ctrl-D (or maybe Ctrl-Z):")
    keyin = inputs()
    vec1 = npFromStrings(keyin)
    vec1 = vec1.flatten()
    # vec length (extrinsic + intrinsic) should be between 21 and 31
    # [2 + 3 + 3 + 9 + 4, 2 + 3 + 3 + 9 + 14]
    if vec1.size > 31:
        vec1 = vec1[0:31]
    if vec1.size < 22:
        _vec1 = np.zeros(22, dtype=float)
        _vec1[0:vec1.size] = vec1[:]
        vec1 = _vec1
    imgSize1 = vec1[0:2].astype(int).flatten()
    rvec1 = vec1[2:5].reshape((3, 1))
    tvec1 = vec1[5:8].reshape((3, 1))
    cmat1 = vec1[8:17].reshape((3, 3))
    dvec1 = vec1[17:].reshape((1, -1))
    
    print("# Enter 22 to 31 reals to initialize the Right camera. These reals are image size (width and height) (2), rvec (3), tvec (3), cmat(9), dvec(4 to 14) and Ctrl-D (or maybe Ctrl-Z):")
    keyin = inputs()
    vec2 = npFromStrings(keyin)
    vec2 = vec2.flatten()
    # vec length (extrinsic + intrinsic) should be between 21 and 31
    # [2 + 3 + 3 + 9 + 4, 2 + 3 + 3 + 9 + 14]
    if vec2.size > 31:
        vec2 = vec1[0:31]
    if vec2.size < 22:
        _vec2 = np.zeros(22, dtype=float)
        _vec2[0:vec2.size] = vec2[:]
        vec2 = _vec2
    imgSize2 = vec2[0:2].astype(int).flatten()
    rvec2 = vec2[2:5].reshape((3, 1))
    tvec2 = vec2[5:8].reshape((3, 1))
    cmat2 = vec2[8:17].reshape((3, 3))
    dvec2 = vec2[17:].reshape((1, -1))

    # get intrinsic parameters of left camera 
    fc_left = np.array([cmat1[0,0], cmat1[1,1]], dtype=float)
    cc_left = np.array([cmat1[0,2], cmat1[1,2]], dtype=float)
    kc_left = dvec1.flatten()[0:5]
    alpha_c_left = 0.0
    r44_1 = np.eye(4, dtype=float)
    r44_1[0:3, 0:3] = cv.Rodrigues(rvec1)[0]
    r44_1[0:3, 3] = tvec1.flatten()[0:3]
    
    # get intrinsic parameters of right camera
    fc_right = np.array([cmat2[0,0], cmat2[1,1]], dtype=float)
    cc_right = np.array([cmat2[0,2], cmat2[1,2]], dtype=float)
    kc_right = dvec2.flatten()[0:5]
    alpha_c_right = 0.0
    r44_2 = np.eye(4, dtype=float)
    r44_2[0:3, 0:3] = cv.Rodrigues(rvec2)[0]
    r44_2[0:3, 3] = tvec2.flatten()[0:3]

    # extrinsic parameters
    r44 = np.matmul(r44_2, np.linalg.inv(r44_1))
    R = r44[0:3, 0:3].copy().reshape(3, 3)
    T = r44[0:3, 3].copy().reshape(3, 1)
    om = cv.Rodrigues(R)[0].reshape(3, 1)
    
    # save to file
    print("# Enter the .mat full path you want to save (e.g., c:\\data\\calib.mat): ")
    file_path = input().strip()
    
    variables = {'fc_left': fc_left, 'cc_left': cc_left, 'kc_left': kc_left, 
                 'alpha_c_left': alpha_c_left, 
                 'fc_right': fc_right, 'cc_right': cc_right, 'kc_right': kc_right, 
                 'alpha_c_right': alpha_c_right,
                 'R': R, 'T': T, 'om': om}   
    scipy.io.savemat(file_path, variables)
    return

    
    