from math import *
from str2argDict import str2argDict
import sys

# This class represents an argument of a float (a real) number. 
# This class is designed for the GUI of the arguments.
# As long as a programmer defines all arguments through this type of class, 
# a GUI of this function can be automatically generated.
# For example, there is a function named lawOfCosine, which calculates 
# the length of the third side of a triangle:
#    def lawOfCosine(a, b, angle):
#        return sqrt(a**2 + b**2 - 2*a*b*cos(angle))
#    The arguments of this function are a, b, and angle.
#    The type of a, b, and angle are all float, while a and b are positive, 
#    and angle is between 0 and pi (180 degrees).
#    You can define the arguments as follows:
#    a = ArgFloat(name='a', desc='The length of the first side of the triangle', min=0, default='1.0')
#    b = ArgFloat(name='b', desc='The length of the second side of the triangle', min=0, default='1.0')
#    angle = ArgFloat(name='angle', desc='The angle between the first and second sides of the triangle', 
#                     min=0, max=pi, default='pi/4')
#    a.desc() returns 'The length of the first side of the triangle'
#    a.value() returns the current value of a as a float.
#    a.set('1.0') sets the value of a to 1.0
#    a.set('1+sqrt(2)') sets the value of a to 1+sqrt(2) as set() uses eval()
#    a.check('1.0') returns (True, "OK")
#    a.check('-1.0') returns (False, "The value is out of the range.")
#    a.check('one thousand') returns (False, "The value is not a float.")
#    a.check() returns the same as a.check(a.value()) 
#    a.min returns 0
#
# Similar classes could include ArgString, ArgImage, ArgArray, ArgString

class ArgFloat:
    # a constructor of an empty object of the class
    def __init__(self):
        self.name = None
        self.desc = None
        self.value = None
        self.default = None
        self.min = None
        self.max = None
        return
    # an alternative constructor of the class given a dictionary
    # For example, arg1 = ArgFloat(); arg1.fromArguments(name='a', desc='Edge 1 of a triangle', default=1, min=0, max=1e30)
    def fromArguments(self, name=None, desc=None, default=None, min=float('-inf'), max=float('inf')):
        # constructor: assigning name
        if type(name) == type(None):
            self.name = None
        elif type(name) == str:
            self.name = name
        else:
            print('# ArgFloat.fromArguments(): Warning: The name of the argument should be a string. Skipped it.')
            self.name = None
        # constructor: assigning desc
        if type(desc) == type(None):
            self.desc = None
        elif type(desc) == str:
            self.desc = desc
        else:
            print('# ArgFloat.fromArguments(): Warning: The description of the argument should be a string. Skipped it.')
            self.desc = None
        # constructor: assigning min
        if type(min) == type(None):
            self.min = None
        elif type(min) == float:
            self.min = min
        elif type(min) == str:
            try:
                self.min = float(eval(min)) # min can be a string that contains an expression
            except:
                print('# ArgFloat.fromArguments(): Warning: The min of the argument should be a float or a string that eval() can handle. Skipped it.')
                self.min = None
        # constructor: assigning max
        if type(max) == type(None):
            self.max = None
        elif type(max) == float:
            self.max = max
        elif type(max) == str:
            try:
                self.max = float(eval(max)) # max can be a string that contains an expression
            except:
                print('# ArgFloat.fromArguments(): Warning: The max of the argument should be a float or a string that eval() can handle. Skipped it.')
                self.max = None
        # constructor: assigning default
        if type(default) == type(None):
            self.default = None
        elif type(default) == str:
            try:
                self.default = float(eval(default)) # default can be a string that contains an expression
            except:
                print('# ArgFloat.fromArguments(): Warning: The default of the argument should be a float or a string that eval() can handle. Skipped it.')
                self.default = None
    # an alternative constructor of the class given a dictionary
    # For example, arg1 = ArgFloat(); arg1.fromDict({'name':'a', 'desc':'Edge 1 of a triangle', 'default':1.0, 'min':0, 'max':1e30})
    def fromDict(self, argDict):
        name = argDict.get('name')
        desc = argDict.get('desc')
        min, max, default = None, None, None
        if 'min' in argDict:
            min = argDict.get('min')
        if 'max' in argDict:
            max = argDict.get('max')
        if 'default' in argDict:
            default = argDict.get('default')
        self.fromArguments(name, desc, default, min, max)
    # an alternative constructor of the class given a string
    # For example, arg1 = ArgFloat(); arg1.fromString('--name a --desc Edge 1 of a triangle --default 1.0 --min 0 --max 1e30')
    def fromString(self, theStr):
        argDict = str2argDict(theStr)
        self.fromDict(argDict)
    # function type is float
    # For example, arg1.dtype() returns 'float'
    # improUiWrap.py uses this function to show the type as a label through lb_type.grid(row=0, column=0)
    def dtype(self):
        return 'float'
    # In many of the cases, this function directly returns the value of self.desc.
    # For example, arg1.desc() returns 'Edge 1 of a triangle'
    # improUiWrap.py uses this function to show the description through lb_desc.grid(row=0, column=2)
    def funcDesc(self):
        return self.desc + " (min: %g, max: %g)" % (self.min, self.max)
    # casting the value to a float
    # This function is called by class functions set() and check()
    # This function does not change the value of this object. 
    # For example, casting(1.0) or casting('1.0') or casting('acos(0.0)') returns 1.0.
    # without changing the value of this object.
    # Normally improUiWrap.py does not call casting().
    def casting(self, value):
        if type(value) == int or type(value) == float:
            to_float = float(value)
            return to_float
        elif type(value) == str:
            try:
                to_float = float(eval(value))
                return to_float
            except:
                print('# ArgFloat.casting(): Warning: The value cannot be evaluated as a float. Skipped it.')
                return nan
        else:
            print('# ArgFloat.casting(): Warning: The value is unknown type. Skipped it.')
            return nan
    # set the value of the argument
    # The value can be a float, an integer, or a string that can be evaluated (by eval()) to a float.
    # This function changes self.value of this object.
    # For example, .set(1.0) or .set('1.0') or .set('acos(0.0)') changes the value of this object to 1.0.
    # improUiWrap.py uses set(text_in_text_entry), then get the value by get(), before calling the major function.
    def set(self, value):
        trial_value = self.casting(value)
        self.value = trial_value
    # get the complete data of this argument 
    # For an ArgFloat object, self.get() is about the same as float(self.value), but for ArgImage 
    # .get() returns the complete image (i.e., 2D array of pixels), while .value is the file path of the image.
    def get(self):
        try:
            if type(self.value) == str:
                return float(eval(self.value))
            return float(self.value)
        except:
            print("# ArgFloat.get(): Error. The value of %s is not a float." % (self.name))
            return None
    # convert the current value into a string
    # This function returns the string form of the current value
    # For example, if the current value is 1.0, this function returns '1.0'.
    def toString(self):
        return str(self.value)
    # check if the value is within the range given a trial_value
    # if trial_value is not given, check the current value
    # improUiWrap.py calls this function, get the returned result (True or False) and the message, 
    # and popups a message box if the message length is not zero.
    def check(self, trial_value=None):
        if trial_value == None:
            trial_value = self.value
        try:
            casted_value = self.casting(trial_value)
        except:
            msg = "# ArgFloat.check(): Error. The value of %s is not a float." % (self.name)
            return (False, msg)
        if casted_value >= self.min and casted_value <= self.max:
            msg = "# ArgFloat.check(): Good. The value of %s (%f) is within the valid range." % (self.name, casted_value)
            return (True, msg)
        else:
            msg = "# ArgFloat.check(): Warning. The value of %s (%f) is not within the valid range." % (self.name, casted_value)
            return (False, msg)

# test code
if __name__ == '__main__':

    # a function that calculates the length of the third side of a triangle
    def lawOfCosine(a, b, angle):
        c = sqrt(a**2 + b**2 - 2*a*b*cos(angle))
        area = 0.5 * a * b * sin(angle)
        return (c, area)
    # import improUiWrap
    from improUiWrap import improUiWrap

    arg1 = ArgFloat()
    arg1.fromString('--name edge1 --desc The length of the first side of the triangle --min 0 --max 1e30 --default 1.0')
    arg2 = ArgFloat()
    arg2.fromString('--name edge2 --desc The length of the second side of the triangle --min 0 --max 1e30 --default 1.0')
    arg3 = ArgFloat()
    arg3.fromString('--name angle --desc The angle between the first and second sides of the triangle --min 0 --max 3.141592653589793 --default 0.7853981633974483')
    inArgList = [arg1, arg2, arg3]
    arg4 = ArgFloat()
    arg4.fromString('--name edge3 --desc The length of the third side of the triangle --min 0 --max 1e30 --default 0.0')
    arg5 = ArgFloat()
    arg5.fromString('--name area --desc The area of the triangle --min 0 --max 1e30 --default 0.0')
    outArgList = [arg4, arg5]
    improUiWrap(inArgList, outArgList, lawOfCosine, winTitle='GUI for law of cosine')
