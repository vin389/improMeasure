import copy

# ArgAbstract is an abstract class that has the following functions:
#   __init__(self): the constructor which initializes basic variables
#                 The basic variables are: .name, .value, .desc, .default
#                .name: the name of the argument (default is '')
#                .desc: the description of the argument (default is '')
#                .default: the default string representation of the argument (default is "")
#                The .value is initialized to None. It is normally set later by user (or, GUI user) ,
#                instead of setting by constructor.
#                For example, 
#                  arg1 = ArgFloat(name='radius', desc='the radius of a circle', default='1.0', min=0.0)
#                  where ArgFloat is a sub-class of ArgAbstract. 
#  
class ArgAbstract:
    # constructor 
    def __init__(self, name, desc='(empty description)', default=None):
        # name
        if type(name) == str:
            self.name = name
        else:
            print("# %s.__init__(): Error: name is supposed to be a string. But got: " % self.__class__.__name__, name)
            print("# %s.__init__(): Error: name is set to an empty string." % self.__class__.__name__)
            self.name = ''
        # desc
        if type(desc) == str:
            self.desc = desc
        else:
            print("# %s.__init__(): Error: desc is supposed to be a string. But got: " % self.__class__.__name__, desc)
            print("# %s.__init__(): Error: desc is set to an empty string." % self.__class__.__name__)
            self.desc = ''
        # value is set to None (uncondifionally)
        self.value = None
    # dtype: the data type of the argument. It is a pure virtual function that should be implemented in the derived class.
    # Sub-classes should implement this function. For example, ArgFloat.dtype() returns "float".
    def dtype(self):
        print("# %s.dtype: Error. dtype() should be implemented." % self.__class__.__name__)
        return type(None)
    # descFunc: function that returns the description
    def descFunc(self):
        return self.desc
    # __str__: the string representation of the argument
    # This function should be implemented in the derived class. If not, it returns the string representation of the self.value
    def __str__(self):
        return str(self.value)
    # toString(): the same as __str__
    def toString(self):
        return self.__str__()
    # setFromString(): set the value from a string
    # This function should be implemented in the derived class. If not, it sets the value to theStr.
    # This function returns (True or False, the error message)
    def setFromString(self, theStr):
        return (False, "# %s.setFromString(): Error: setFromString() should be implemented." % self.__class__.__name__)
    # set(): set the value from a string or a value
    # Normally for a sub-class, if the setFromString() is implemented, the set() does not need to be implemented.
    # The value can be a string or a value in its original data type (e.g., image, array, etc.)
    # Note that the set() may use shallow copy. If you want to use deep copy, you should call set(a.copy()) instead of set(a).
    def set(self, value):
        if type(value) == str:
            return self.setFromString(value)
        else:
            self.value = value
            return (True, "")
    # get(): get (a clone) of the value of this object
    # If necessary, it gets the data of the clone from a file. For example, ArgImage.get() reads the image from a file.
    def get(self):
        return copy.deepcopy(self.value)
    # display(): display the value of the argument
    # For ArgImage, it displays the image in a window, by using cv2.imshow() or improMeasure.imshow2() and returns "". 
    # For ArgArray it returns self.value.__str__() (so that it can display its short form in a message box)
    # For ArgFloat it returns self.value.__str__() and the upper/lower bounds check result. 
    # If not defined, it returns self.value.__str__() (so, ArgArray does not need to implement this function)
    def display(self):
        return self.__str__()
    # check() checks the value of the argument that is written in the text entry.
    # This function sets the self.value from the entry_text, and then checks the self.value.
    # If entry_text is not given, it directly processes the self.value.
    # This function returns (True/False, the message of the checking result)
    def check(self, entry_text=None):
        if entry_text is not None:
            (result, msg) = self.setFromString(entry_text)
            if result == False:
                return (False, msg)            
        # check or display the value

    

    
    



        
    
    
    