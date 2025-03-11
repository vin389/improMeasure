import numpy as np
import matplotlib.pyplot as plt
from ArgArray import ArgArray
from improUiWrap import improUiWrap
from npFromStr import npFromStr

class ArgTimeseries(ArgArray):
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
            # if it is a csv or txt file, read data from the file
            try:
                theArray = npFromStr('file ' + entry_text)
                theArray = theArray.reshape(self.dimension)
            except:
                print('# ArgTimeseries.check(): Warning: Tried to load the file %s, but failed. Skipped it.' % entry_text)
                return (False, "The input is not a valid array.")
        else:
            # if it is not csv nor txt, convert the string to a numpy array
            # then print the array to the console
            try:
                theArray = npFromStr(entry_text)
                theArray = theArray.reshape(self.dimension)
            except:
                print('# ArgArray.check(): Warning: Tried to convert the string %s to a numpy array, but failed. Skipped it.' % entry_text)
                return (False, "The input is not a valid array.")
        # Display the time series in a plot
        # The first column is the x-axis, and the rest are y-axis
        # Use small square markers for all points
        try:
            # if theArray has only one column, plot it as a line plot
            if theArray.shape[1] == 1:
                plt.plot(theArray[:,0], 's-')
                plt.xlabel('Time Step')
            # if theArray has two or more columns, say, N colums, plot N-1 columns as y-axis
            else:
                plt.plot(theArray[:,0], theArray[:,1:], 's-')
            plt.xlabel('Time')
            return (True, "")
        except:
            print('# ArgTimeseries.check(): Warning: Tried to plot the array, but failed. Skipped it.')
            return (False, "Warning: Tried to plot the array, but failed. Skipped it.")

# test code
if __name__ == '__main__':

    # a function that calculates the length of the third side of a triangle
    def plusOne(mat):
        return np.array(mat + 1.0)
    from improUiWrap import improUiWrap

    arg1 = ArgTimeseries()
    arg1.fromString('--name src --desc The source array --dimension -1 3 --default 1 2 3 4 5 6')
    inArgList = [arg1]
    arg2 = ArgTimeseries()
    arg2.fromString('--name dst --desc The processed array --dimension -1 3 --default 1 2 3 4 5 6')
    outArgList = [arg2]
    improUiWrap(inArgList, outArgList, plusOne, winTitle='GUI for ArrayPlusOne')    