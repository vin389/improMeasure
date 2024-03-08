import numpy as np

def npFromStr(theStr: str, dtype=float):
    """
    This function converts a string to 1D numpy ndarray.
    Separators like {, }, [, ], (, ), ;, \", \', \t, \n, are converted to a space
    Nans like n/a, #N/A, na, are converted to nan (indicating np.nan)
    Multi-dimensional representation are converted to 1 dimension. 
    For example:
        npFromStr('1 2 3') ==> [1., 2., 3.]
        npFromStr('[[1, 2], [nan, 4]]') ==> [1., 2., nan, 4.]
        npFromStr('file c:/test/test.csv') ==> (read the file by np.loadtxt)

    Parameters
    ----------
    theStr : str
        the string to be converted to a 1D numpy array.
    dtype : dtype, optional
        Data type to convert to. The default is float.

    Returns
    -------
    np.ndarray
        the 1D np.array converted from the theStr.

    """
    # Check if the string starts with 'file '
    if type(theStr) == str and theStr.strip().split()[0].lower() == 'file':
        mat = np.loadtxt(theStr.strip().split()[1])
        return mat
    
    if type(theStr) == str:
        _str = theStr.strip()
    elif type(theStr) == list or type(theStr) == tuple:
        return npFromStrs(theStr)
    else: 
        _str = str(theStr)
#
    if type(_str) == str:
        _str = theStr
        _str = _str.replace('\"', ' ').replace('\'', ' ')
        _str = _str.replace('\t', ' ').replace('\n', ' ')
        _str = _str.replace(',', ' ').replace(';', ' ')
        _str = _str.replace('[', ' ').replace(']', ' ')
        _str = _str.replace('{', ' ').replace('}', ' ')
        _str = _str.replace('(', ' ').replace(')', ' ')
        _str = _str.replace(']', ' ').replace('na ', 'nan').replace('\n',' ')
        _str = _str.replace('n/a', 'nan').replace('#N/A', 'nan')
        mat = np.fromstring(_str, dtype=float, sep=' ').astype(dtype)
    return mat


def npFromStrs(theStrList: list):
    """
    Converts a list of strings (which only contains floats) to a numpy 
    float array (in 1D). The separator can be ',', ' ', '\t', '\n', 
    '(', ')', '[', ']', '{', '}'. The 'nan' or 'na' would be considered 
    as np.nan. 
    The returned numpy will be in 1D. 
    For example:
        npFromStrs(['1.2 , 2.3', 
                       'nan \n 4.5'])
            returns array([1.2, 2.3, nan, 4.5])
    """
    theType = type(theStrList)
    if theType == str:
        return npFromStr(theStrList.strip())
#   elif theType == list or theType == tuple:
    else: 
        # assuming the type is list or tuple, or 
        # other types that runs with the following for loop
        _str = ''
        for i in theStrList:
            _str += (str(i).strip() + '\n')
        return npFromStr(_str)


if __name__ == '__main__':
    while(True):
        print("# Enter a string to represent np.array (or enter QUIT or END to quit):")
        ipt = input()
        if ipt.lower() == 'quit' or ipt.lower() == 'end':
            break
        mat = npFromStr(ipt)
        print("Your input is converted to a np.array:")
        print("  shape: ", mat.shape)
        print("  ", mat)
        