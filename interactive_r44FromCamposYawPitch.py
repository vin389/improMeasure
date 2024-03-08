from math import cos, sin, pi
import numpy as np
from r44FromCamposYawPitch import r44FromCamposYawPitch




def interactive_r44FromCamposYawPitch(arg=''):
    print("# This function calculates the 4-by-4 matrix form of extrinsic ")
    print("# parameters of a camera according to the camera position, the")
    print("# yaw, and the pitch.")
    print("# If you continue, you will be asked to enter:")
    print("#   1. Camera position (3 reals):")
    print("#   2. Yaw (1 real, in degrees):")
    print("#   3. Pitch (1 real, in degrees):")
    print("#   4. Fullpath of the output 4-by-4 matrix file in numpy format:")
    print("# For example:")
    print("#   0.0  -1.732  0.0  ")
    print("#   30  ")
    print("#   10  ")
    print("# You ")
    print("#")
    togo = input("# Do you want to continue? Enter Y or y to continue, otherwise to return.\n")
    if togo != 'Y' and togo != 'y':
        print("# Returning back.")
        return
    campos_str = input("# Enter camera position (3 reals): ")
    




def npFromString(theStr):
    if type(theStr) != str:
        return np.array([])
    _str = theStr
    _str = _str.replace(',', ' ').replace(';', ' ').replace('[', ' ')
    _str = _str.replace(']', ' ').replace('na ', 'nan').replace('\n',' ')
    _str = _str.replace('n/a', 'nan').replace('#N/A', 'nan')
    mat = np.fromstring(_str, dtype=float, sep=' ')
    return mat


def input2(prompt=""):
    """
    This function is similar to Python function input() but if the returned
    string starts with a hashtag (#) this function ignores the line of the
    strin and runs the input() function again.
    The head spaces and tail spaces are removed as well.
    This function only allows user to edit a script for a series of input,
    but also allows user to put comments by starting the comments with a
    hashtag, so that the input script is earier to understand.
    For example, a BMI converter could run in this way:
    /* -------------------------------
    1.75  (user's input)
    70    (user's input)
    The BMI is 22.9
    --------------------------------- */
    The user can edit a file for future input:
    /* ---------------------------------
    # This is an input script for a program that calculates BMI
    # Enter height in unit of meter
    1.75
    # Enter weight in unit of kg
    70

    Parameters
        prompt  A String, representing a default message before the input.
    --------------------------------- */
    """
    theInput = ""
    if len(prompt) == 0:
        thePrompt = ""
    else:
        thePrompt = "# " + prompt
        print(thePrompt)
    # run the while loop of reading
    while(True):
        theInput = input()
        theInput = theInput.strip()
        if (len(theInput) == 0):
            continue
        if (theInput[0] == '#'):
            continue
        break
    # remove everything after the first #
    if theInput.find('#') >= 0:
        theInput = theInput[0:theInput.find('#')]
    return theInput


if __name__ == '__main__':
    interactive_r44FromCamposYawPitch()
    