from math import *
from str2argDict import str2argDict
from npFromStr import npFromStr
import sys
import numpy as np

# This class represents an argument of an array, stored by numpy array. 
# The array is basically a float, relatively small, np.ndarray. 
# It is not designed for a large array, such as an image, as it loads and saves an array as a csv file.
# This class is designed for the GUI of the arguments.
# As long as a programmer defines all arguments through this type of class, 
# a GUI of this function can be automatically generated.
# Similar classes could include ArgInt, ArgString, ArgImage, ArgMatrix, ArgString
# The class is used by improUiWrap.py, which is a wrapper for a function that has a GUI.

class ArgArray:
    # a constructor of an empty object of the class
    def __init__(self):
        self.name = None       # For example, 'warp_matrix'
        self.desc = None       # For example, 'A 2x3 matrix for affine or 3x3 matrix for perspective transformation'
        self.value = None      # For example, '1 0 0   0 1 0'
        self.default = None    # For example, '1 0 0   0 1 0'
        self.dimension = [-1]  # For example, [2,3] or [3,3] or [-1,3] (see Numpy dimension for meaning of -1)
        self.path = None       # For example, 'examples/warp_matrix.csv'
        return
    # an alternative constructor of the class given a dictionary
    # For example, arg1 = ArgArray(); arg1.fromArguments(name='warp_matrix', desc='A 2x3 matrix for affine transformation', 
    #              default='1 0 0   0 1 0', dimension='2 3', path='examples/warp_matrix.csv')
    def fromArguments(self, name=None, desc=None, default=None, dimension=None, path=None):
        # constructor: assigning name
        if type(name) == type(None):
            self.name = None
        elif type(name) == str:
            self.name = name
        else:
            print('# ArgArray.fromArguments(): Warning: The name of the argument should be a string. Skipped it.')
            self.name = None
        # constructor: assigning desc
        if type(desc) == type(None):
            self.desc = None
        elif type(desc) == str:
            self.desc = desc
        else:
            print('# ArgArray.fromArguments(): Warning: The description of the argument should be a string. Skipped it.')
            self.desc = None
        # constructor: assigning default
        if type(default) == type(None):
            self.default = None
        elif type(default) == str:
            self.default = default
        else:
            print('# ArgArray.fromArguments(): Warning: The default of the argument should be a string. Skipped it.')
            self.desc = None
        # constructor: assigning dimension
        if type(dimension) == type(None):
            self.dimension = [-1]
        elif type(dimension) == np.ndarray:
            self.dimension = np.array(dimension, dtype=int)
        elif type(dimension) == str:
            self.dimension = npFromStr(dimension, dtype=int)
        else:
            print('# ArgArray.fromArguments(): Warning: The dimension of the argument should be a string. Skipped it.')
            self.desc = None
        # constructor: assigning path
        if type(path) == type(None):
            self.path = None
        elif type(path) == str:
            self.path = path
        else:
            print('# ArgArray.fromArguments(): Warning: The path of the argument should be a string. Skipped it.')
            self.path = None
        
    # an alternative constructor of the class given a dictionary
    # For example, arg1 = ArgFloat(); arg1.fromDict({'name':'a', 'desc':'Edge 1 of a triangle', 'default':1.0, 'min':0, 'max':1e30})
    def fromDict(self, argDict):
        name = argDict.get('name')
        desc = argDict.get('desc')
        default, dimension, path = None, None, None
        if 'default' in argDict:
            default = argDict.get('default')
        if 'dimension' in argDict:
            dimension = argDict.get('dimension')
        if 'path' in argDict:
            path = argDict.get('path')
        self.fromArguments(name, desc, default, dimension, path)
    # an alternative constructor of the class given a string
    # For example, arg1 = ArgFloat(); arg1.fromString('--name a --desc Edge 1 of a triangle --default 1.0 --min 0 --max 1e30')
    def fromString(self, theStr):
        argDict = str2argDict(theStr)
        self.fromDict(argDict)
    # function type is float
    # For example, arg1.dtype() returns 'float'
    # improUiWrap.py uses this function to show the type as a label through lb_type.grid(row=0, column=0)
    def dtype(self):
        return 'array'
    # In many of the cases, this function directly returns the value of self.desc.
    # For example, arg1.desc() returns 'Edge 1 of a triangle'
    # improUiWrap.py uses this function to show the description through lb_desc.grid(row=0, column=2)
    def funcDesc(self):
        return self.desc + " (dim: " + str(self.dimension) + ")"
    # casting the entry string to a float
    # This function is called by class functions set() and check()
    # This function does not change the value of this object. 
    # For example, casting('1 2 3') returns np.array([1.0, 2.0, 3.0])
    # For example, casting('file.txt') or casting('file.csv') or casting('file.npy') returns the content of the file.
    #     The extension must be one of the following: .txt or .csv. The .txt and .csv are comma-separated values. 
    # Normally improUiWrap.py does not call casting().
    def casting(self, value):
        if type(value) == np.ndarray:
            return value.reshape(self.dimension)
        if type(value) == str:
            value = value.strip()
            if value[-4:] == '.csv' or value[-4:] == '.txt' or value[-4:] == '.npy':
                try:
                    theArray = npFromStr('file ' + value)
                    return theArray.reshape(self.dimension)
                except:
                    print('# ArgArray.casting(): Warning: Tried to load the file %s, but failed. Skipped it.' % value)
                    return None
            else:
                try:
                    theArray = npFromStr(value)
                    return theArray.reshape(self.dimension)
                except:
                    print('# ArgArray.casting(): Warning: Tried to convert the string %s to a numpy array, but failed. Skipped it.' % value)
                    return None
        else:
            print('# ArgArray.casting(): Warning: The value is unknown type. Skipped it.')
            return None
    # set the value of the argument
    # The value can be a float, an integer, or a string that can be evaluated (by eval()) to a float.
    # This function changes self.value of this object.
    # For example, .set(1.0) or .set('1.0') or .set('acos(0.0)') changes the value of this object to 1.0.
    # improUiWrap.py uses set(text_in_text_entry), then get the value by get(), before calling the major function.
    def set(self, value):
        trial_value = self.casting(value)
        self.value = trial_value
    # get the complete data of this argument 
    # It is called by improUiWrap.py before calling the major function.
    def get(self):
        if type(self.value) == np.ndarray and self.value.size > 0:
            return self.value.reshape(self.dimension)
        else:
            self.load()
            return self.value
    # convert the current value into a string
    def toString(self):
        if type(self.value) == np.ndarray and self.value.size > 0:
            return str(self.value.flatten())
    # load the numpy array from the file self.path
    # This function load data from file, and change the value of this object.
    # Then it returns the value.
    # For example, if self.path is 'examples/warp_matrix.csv', load() reads the file, saves the array 
    # self.value, and returns the content of the file.
    def load(self, filePath=None):
        if filePath == None:
            filePath = self.path
        self.value = self.casting(filePath)
        return self.value
    # save the current self.value to the file self.path
    # It is called by improUiWrap.py after the major function is called.
    def save(self, filePath=None):
        if filePath == None:
            filePath = self.path
        if type(self.value) == np.ndarray:
            if filePath.endswith('.csv'):
                np.savetxt(filePath, self.value, delimiter=',')
            elif filePath.endswith('.txt'):
                np.savetxt(filePath, self.value, delimiter=' ')
            elif filePath.endswith('.npy'):
                np.save(filePath, self.value)
        return
    # check the value of the argument that is written in the text entry
    # If the text entry ends with '.txt', '.csv', or '.npy' it tries to open the file by default the default text editor.
    # .txt is considered as a space-separated values, 
    # .csv is considered as a comma-separated values.
    # .npy is considered as a numpy format.
    # Otherwise, it tries to convert the string into a numpy array by npFromStr, then reshape to the self.dimension.
    # For .txt or .csv file, it displays the content of the file in a text editor.
    # Otherwise, directly print the value on the console.
    def check(self, entry_text):
        # Convert string (entry_text) to a numpy array
        if entry_text[-4:] == '.csv' or entry_text[-4:] == '.txt':
            # if it is a csv or txt file, read data from the file, and open the file in a text editor
            try:
                theArray = npFromStr('file ' + entry_text)
                theArray = theArray.reshape(self.dimension)
                # open the file by the default text editor
                import os
                os.system('start ' + entry_text)
                return (True, "")
            except:
                print('# ArgTimeseries.check(): Warning: Tried to load the file %s, but failed. Skipped it.' % entry_text)
                return (False, "The input is not a valid array.")
        else:
            # if it is not csv nor txt, convert the string to a numpy array
            # then print the array to the console
            try:
                theArray = npFromStr(entry_text)
                theArray = theArray.reshape(self.dimension)
                print('# The array is:', theArray)
                return (True, "")
            except:
                print('# ArgArray.check(): Warning: Tried to convert the string %s to a numpy array, but failed. Skipped it.' % entry_text)
                return (False, "The input is not a valid array.")


# test code
if __name__ == '__main__':

    # a function that calculates the length of the third side of a triangle
    def plusOne(mat):
        return np.array(mat + 1.0)
    from improUiWrap import improUiWrap

    arg1 = ArgArray()
    arg1.fromString('--name src --desc The source array --dimension 2 3 --default 1 2 3 4 5 6')
    inArgList = [arg1]
    arg2 = ArgArray()
    arg2.fromString('--name dst --desc The processed array --dimension 2 3 --default 1 2 3 4 5 6')
    outArgList = [arg2]
    improUiWrap(inArgList, outArgList, plusOne, winTitle='GUI for ArrayPlusOne')
