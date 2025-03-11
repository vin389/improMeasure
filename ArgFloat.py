from ArgAbstract import ArgAbstract
import copy 
# ArgFloat: a class for a float argument
# ArgFloat is a sub-class of ArgAbstract
class ArgFloat(ArgAbstract):
    # constructor
    def __init__(self, name='', desc='(empty description)', default=None, \
                 min=float('-inf'), max=float('inf')):
        # call the constructor of the base class
        super().__init__(name, desc, default)
        # min: the minimum value of the argument
        self.min = min
        # max: the maximum value of the argument
        self.max = max
    # dtype: the data type of the argument
    def dtype(self):
        return 'float'
    # descFunc: function that returns the description
    def descFunc(self):
        return self.desc + ' (valid range: [%f, %f])' % (self.min, self.max)
    # __str__: the string representation of the argument
    # This function should be implemented in the derived class. If not, it returns the string representation of the self.value
    def __str__(self):
        return str(self.value)
    # toString(): the same as __str__
    def toString(self):
        return self.__str__()
    # setFromString: set the value from a string
    # This function returns (True or False, the error message)
    def setFromString(self, theStr):
        try:
            # convert the string to a float
            self.value = float(theStr)
            # check the value is in the range of [min, max]
            if self.min is not None and self.value < self.min:
                return (False, "# ArgFloat.setFromString(): Error: The value %f is smaller than the minimum value %f." % (self.value, self.min))
            if self.max is not None and self.value > self.max:
                return (False, "# ArgFloat.setFromString(): Error: The value %f is larger than the maximum value %f." % (self.value, self.max))
            return (True, "")
        except:
            return (False, "# ArgFloat.setFromString(): Error: The input is not a valid float.")
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
        if self.value < self.min:
            return (False, "# ArgFloat.check(): The %s value %f is smaller than the minimum value %f." % (self.name, self.value, self.min))
        if self.value > self.max:
            return (False, "# ArgFloat.check(): The %s value %f is larger than the maximum value %f." % (self.name, self.value, self.max))
        return (True, "# ArgFloat.check(): The %s value %f is valid." % (self.name, self.value))
    
