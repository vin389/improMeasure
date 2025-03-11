from math import *
from str2argDict import str2argDict
import sys, os
import cv2
import numpy as np
import imshow2

# This class represents an argument of an image.
# This class is designed for the GUI of the arguments.
# As long as a programmer defines all arguments through this type of class, 
# a GUI of this function can be automatically generated.

class ArgImage:
    # a constructor of an empty object of the class
    def __init__(self):
        self.name = None
        self.desc = None
        self.value = None
        self.path = None
        self.default = None
        return
    # an alternative constructor of the class given arguments
    # For example, argImage.fromArguments(name='src', desc='original image', default='c:/temp/IMG_0000.JPG')
    def fromArguments(self, name=None, desc=None, default=None):
        # constructor: assigning name
        if type(name) == type(None):
            self.name = None
        elif type(name) == str:
            self.name = name
        else:
            print('# ArgImage.fromArguments(): Warning: The name of the argument should be a string. Skipped it.')
            self.name = None
        # constructor: assigning desc
        if type(desc) == type(None):
            self.desc = None
        elif type(desc) == str:
            self.desc = desc
        else:
            print('# ArgImage.fromArguments(): Warning: The description of the argument should be a string. Skipped it.')
            self.desc = None
        # constructor: assigning default
        if type(default) == type(None):
            self.default = None
        elif type(default) == str:
            self.default = default
        else:
            print('# ArgImage.fromArguments(): Warning: The default value of the argument should be a string. Skipped it.')
            self.default = None
        return
    # an alternative constructor of the class given a dictionary
    # For example, argImage.fromDict({'name':'src', 'desc':'original image', 'default':'c:/temp/IMG_0000.JPG'})
    def fromDict(self, argDict):
        name = argDict.get('name')
        desc = argDict.get('desc')
        default = None
        value = None
        if 'default' in argDict:
            default = argDict.get('default')
        self.fromArguments(name=name, desc=desc, default=default)
    # an alternative constructor of the class given a string
    # For example, argImage.fromString('--name src --desc original image --default c:/temp/IMG_0000.JPG')
    def fromString(self, argStr):
        argDict = str2argDict(argStr)
        self.fromDict(argDict)
        return
    # function type is image
    # For example, arg1.dtype() returns 'image'
    # improUiWrap.py uses this function to show the type as a label through lb_type.grid(row=0, column=0)
    def dtype(self):
        return 'image'
    # In many cases, this function directly returns the value of self.desc.
    # For example, arg1.desc() returns 'Edge 1 of a triangle'
    # improUiWrap.py uses this function to show the description through lb_desc.grid(row=0, column=2)
    def funcDesc(self):
        return self.desc
    # set the value of the argument
    # The value can be a string (of the file path) or an image itself (a 2D numpy array representing an image).
    # This function changes self.value of this object.
    # If the given argument is a string, it is assumed to be the file path, and this function changes self.path 
    # to the given string (assumed to be the path), and reset the self.value (the ndarray) to None. 
    # If the given argument is an image (2D array), it changes the value of this object to a deep copy of the 2D array, 
    # and does not change the self.path because it could be the path that it will save the image to.
    # For example, arg1.set('c:/temp/IMG_0000.JPG') changes the path 'c:/temp/IMG_0000.JPG'
    # For example, arg1.set(cv2.imread('c:/temp/IMG_0000.JPG')) changes the value to a 2D numpy array representing the image.
    # improUiWrap.py uses set(text_in_text_entry), then get the value by get(), before calling the major function.
    def set(self, value):
        if type(value) == str:
            self.value = None
            self.path = value
            print("# ArgImage.set(): The path of the image is set to " + self.path)
        elif type(value) == np.ndarray:
            self.value = value.copy()
            print("# ArgImage.set(): The value of the argument is set to an image, which dimension is ", self.value.shape)
        else:
            print("# ArgImage.set(): Warning: The value of the argument should be a string or an image. Skipped it.")
    # convert the current value into a string
    # This function returns the string form of the current value
    # As this object acquire image data only when it is needed, most of the time the data itself is
    # the file path of the image.
    def toString(self):
        if type(self.path) == str:
            return self.path
        else:
            return ""
    # For ArgImage, check(value) displays the image by using imshow2()
    # if value is not given, check the current value
    # improUiWrap.py calls this function, get the returned result (True or False) and the message, 
    # and popups a message box if the message length is not zero.
    # This function displays the image using imshow2() function.    
    def check(self, value=None, winmax=(1400,700)):
        msg = ""
        if type(value) == type(None) and type(self.value) == np.ndarray and self.value.size > 0:
            # display current image in self.value
            img = self.value
            win_title = "Image (current value) (%dx%dx%d)" % (img.shape[1], img.shape[0], img.shape[2])
            imshow2(win_title, img, winmax=winmax)
            return (True,"")
        if type(value) == type(None) and type(self.path) == str:
            # load image from the file
            img = cv2.imread(self.path)
            if type(img) == np.ndarray and img.size > 0:
                win_title = "Image:%s (%dx%dx%d)" % (self.path, img.shape[1], img.shape[0], img.shape[2])
                imshow2(win_title, img, winmax=winmax)
                return (True, "")
            else:
                return (False, "ArgImage.check(): Cannot read image from the file %s." % self.path)
        if type(value) == str:
            img = cv2.imread(value)
            if type(img) == np.ndarray and img.size > 0:
                win_title = "Image:%s (%dx%dx%d)" % (value, img.shape[1], img.shape[0], img.shape[2])
                imshow2(win_title, img, winmax=winmax)
                return (True, "")
            else:
                return (False, "ArgImage.check(): Cannot read image from the file %s." % value)
        if type(value) == np.ndarray and value.size > 0:
            img = value
            win_title = "Image: given image (%dx%dx%d)" % (img.shape[1], img.shape[0], img.shape[2])
            imshow2(win_title, img, winmax=winmax)
            return (True, "")
        else:
            return (False, "ArgImage.check(): The value of the argument is not an image.")
    # get the complete data of this argument 
    # For an ArgImage object, self.get() returns the value (as an image) (i.e., 2D array of pixels).
    # If the self.value is not an image, this function tries to load the image from self.path.
    # .get() returns the complete image (i.e., 2D array of pixels), while .value is the file path of the image.
    def get(self):
        if type(self.value) == np.ndarray and self.value.size > 0:
            return self.value
        else:
            self.load()
            return self.value
    # load from file
    # If filePath is given, this function load an image from the file, and save the image to the self.value.
    # It does not change self.path. It only loads data from file and saves it to self.value.
    # If filePath is not given, this function load an image from the self.path, and save the image to the self.value.
    def load(self, filePath=None):
        # read the image file through cv2.imread(). If it fails, return None.
        img = None
        if type(filePath) == type(None):
            filePath = self.path
        if type(filePath) == str:
            img = cv2.imread(filePath)
            if type(img) == np.ndarray and img.size > 0:
                self.value = img
                print("# ArgImage.load(): The image is loaded from " + filePath)
            else:
                print("# ArgImage.load(): Warning: Cannot read image from the file %s." % filePath)
        else:
            print("# ArgImage.load(): Warning: The file path should be a string. Skipped it.")        
        return img
    # save image to file
    # This function saves the image (value) to the given file path.
    # This function does not change self.value nor self.path.
    # This function only saves self.value to the given file path.
    # If the filePath is not given, it saves the image to the self.path.
    # If the image can be written to the file path, it prints a message and returns True.
    # If the value is not an image, it prints a warning message and returns False.
    # If it fails to save the image to the file, it prints a warning message and returns False.
    def save(self, filePath):
        if type(filePath) == type(None):
            filePath = self.path
        if type(self.value) == np.ndarray and self.value.size > 0:
            save_ok = cv2.imwrite(filePath, self.value)
            if save_ok == True:
                print("# ArgImage.save(): The image is saved to " + filePath)
            else:
                print("# ArgImage.save(): Warning: Cannot save the image to the file %s." % filePath)
            return False
        else:
            print("# ArgImage.save(): Warning: The value of the argument is not an image. Skipped it.")
            return False


# test code
if __name__ == '__main__':
    # example 1
    from improUiWrap import improUiWrap
    from imshow2 import imshow2
    default_path = 'D:/yuansen/ImPro/improMeasure/examples/2022brb2c56/calibration_cboard_1/G0012011.JPG'
    arg1 = ArgImage()
    arg1.fromArguments(name='src', desc='original image', default=default_path)
    inArgList = [arg1]
    outArgList = []
    improUiWrap(inArgList, outArgList, lambda x: imshow2("Test", x), winTitle='GUI for function')

    # example 2
    from improUiWrap import improUiWrap
    from ArgFloat import ArgFloat
    default_path = 'D:/yuansen/ImPro/improMeasure/examples/2022brb2c56/calibration_cboard_1/G0012011.JPG'
    arg1 = ArgImage()
    arg1.fromArguments(name='src', desc='original image: You can open the file dialog and select the file.', default=default_path)
    arg2 = ArgFloat()
    arg2.fromArguments(name='scale', desc='scale factor', default='0.5')
    inArgList = [arg1, arg2]
    arg3 = ArgImage()
    arg3.fromArguments(name='dst', desc='resized image: You can open the file dialog and select the file to save. After you run the function, the output image will save to the file you select.', default='d:/temp/resized.jpg')
    outArgList = [arg3]
    improUiWrap(inArgList, outArgList, lambda img, scale: (cv2.resize(img, None, fx=scale, fy=scale),), winTitle='GUI for function')

