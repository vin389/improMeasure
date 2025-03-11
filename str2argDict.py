# This function converts a (long) string that describes arguments to a dictionary.
# For example, from 
#     "--type int --name gaussFiltSize --desc Gaussian filter size --min 1 --max 99"
# to 
#    {'type': 'int', 
#     'name': 'gaussFiltSize', 
#     'desc': 'Gaussian filter size', 
#     'min': '1', 
#     'max': '99'}
# Input: 
#   theStr: a string that describes arguments. 
#           For example, "--type int --name gaussFiltSize --desc Gaussian filter size --min 1 --max 99"
# Output:
#   theDict: a dictionary that contains the arguments.
#           For example, {'type': 'int', 'name': 'gaussFiltSize', 
#                         'desc': 'Gaussian filter size', 'min': '1', 'max': '99'}
# Note:
#   1. The string should start with a '--'. For example, '--type int --name gaussFiltSize'
#   2. The key and value should be written as '--key value'. For example, '--max 99'
#   3. The value can be multiple words. For example, '--desc Gaussian filter size'
#   4. The separator is ' --' (i.e., a space followed by two hyphens). 
#      That means ' --' cannot be contained in the value.
#      For example, '--desc a range from A to B' is correct, but '-desc a range from A --B' makes B a key.
#   5. The first key can start with either '--' or ' --'. For example, '--type int' or ' --type int' are both correct.
def str2argDict(theStr):
    # Split the string into a list of words. For example, 
    # "--type int --name gaussFiltSize --desc Gaussian filter size --min 1 --max 99" to 
    # ['type int', 'name gaussFiltSize', 'desc Gaussian filter size', 'min 1', 'max 99']
    theStr = theStr.replace('\t', ' ').replace('\n', ' ').replace('\r', ' ')
    words = theStr.strip().split(sep=' --')
    # If the first letter of the first string is '--', remove it. It is normally '--type'
    if words[0][0] == '-' and words[0][1] == '-':
        words[0] = words[0][2:]
    # Create a dictionary
    theDict = {}
    # Loop through the words
    for i in range(len(words)):
        thisString = words[i] # For example, thisString could be 'desc Gaussian filter size'
        try:
            # split thisString into two words. For example, 'desc Gaussian filter size' to ['desc', 'Gaussian filter size']
            keyAndValue = thisString.split(sep=' ', maxsplit=1)
            # append the first word to the dictionary as a key, and the rest word(s) as the value
            # for example, theDict['desc'] = 'Gaussian filter size'
            theDict[keyAndValue[0]] = keyAndValue[1]
        except:
            # 
            print('# str2argDict(): Warning: Cannot process argument %s. Skipped it.' % thisString)
    return theDict


# Test the function
if __name__ == '__main__':
    # ask the user to input a string and display the dictionary
    # 
    while True:
        print("# " + "-" * 60)
        print('# This function converts a (long) string that describes arguments to a dictionary.')
        print('# For example, "--type int --name gaussFiltSize --desc Gaussian filter size" ')
        print('# Or enter "exit" to exit.')
        theStr = input('# Enter a string:\n')
        # convert theStr to all lower case and remove the leading and trailing spaces
        if theStr.strip().lower() == 'exit':
            break
        try:
            theDict = str2argDict(theStr)
            print(theDict)
        except:
            print('# Error: Cannot convert the string to a dictionary.')
    print('# Done.')

