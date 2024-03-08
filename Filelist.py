import os
import cv2 as cv
import glob


def strToFilelist(ufiles: str):
    """
    Example 1: single file

        strToFilelist('d:/path/DSC00001.JPG') 
            returns a list of a single file name, 
            i.e., ['d:/path/DSC00001.JPG']
            
    Example 2: using wildcard string of * or ?

        strToFilelist('d:/path/DSC*.JPG')
            returns a list of files that satisfy the path with wildcard
        strToFilelist('d:/path/DSC????0.JPG')
            returns a list of files that satisfy the path with wildcard
        ? represents a single letter, while * represents any number of letter(s)
        See https://en.wikipedia.org/wiki/Wildcard_character for details.
        
    Example 3: using c-format %d, start, count

        strToFilelist('d:/path/DSC%05d.JPG, 5, 10')
            returns ['d:/path/DSC00005.JPG', ..., 'd:/path/DSC00014.JPG'], 
            totally 10 files.

    Example 4: using c-format %d, start, count, step

        strToFilelist('d:/path/DSC%05d.JPG, 5, 3, 2')
            returns ['d:/path/DSC00005.JPG', ..., 'd:/path/DSC00009.JPG'], 
            totally 3 files.

    Example 5: using recursive search (using **)

        strToFilelist('d:/path/**/DSC00*.JPG')
            returns a list of files that satisfy the path with wildcard
            When using **, the recursive search is enabled.
            See glob manual for the double asterisk definition.
    """
    if ufiles.find('*') >= 0 or ufiles.find('?') >= 0:
        # if argument contains %, it is confusing. It gives warning and returns [].
        if ufiles.find('%') >= 0:
            print("# Error: strToFilelist(): Do not use % (c-format) and wildcard (* or ?) at the same time. It would be confusing. Now it is returning an empty file list.")
            return []
        # Use glob. If argument string contains double asterisks (**) enable recursive.
        if ufiles.find('**') >= 0:
            files = glob.glob(ufiles, recursive=True)
        else:           
            files = glob.glob(ufiles)
        return files
    elif ufiles.find('%') >= 0:
        ufiles_split = ufiles.split(',')
        if len(ufiles_split) == 3:
            cfiles = ufiles_split[0].strip()
            cfilesStart = int(ufiles_split[1])
            cfilesCount = int(ufiles_split[2])
            files = []
            for i in range(cfilesCount):
                files.append(cfiles % (i + cfilesStart))
            return files
        elif len(ufiles_split) == 4:
            cfiles = ufiles_split[0].strip()
            start = int(ufiles_split[1])
            count = int(ufiles_split[2])
            step= int(ufiles_split[3])
            files = []
            for i in range(start, start + step * count, step):
                files.append(cfiles % (i))
            return files
        else:
            return []
    else:
        # since ufiles have neither % nor wildcard(*,?), it could be 
        # just the file name of a single file
        return [ufiles]

# def find_files(root_dir, extension=''):
#     """
#     Finds recursively all files with a specified extension under a given directory and its subdirectories,
#     and return a list of their file paths, sorted by modification time in ascending order.

#     Parameters
#     ----------
#     root_dir : str
#         The root directory to start the search from.
#     extension : str, optional
#         The file extension to search for. The search is case insensitive.

#     Returns
#     -------
#     file_list : list[str]
#         A list of file paths that match the specified extension, sorted by modification time in ascending order..

#     Example
#     -------
#     find_files('/path/to/dir', 'jpg')
#     ['/path/to/dir/subdir/image1.jpg', '/path/to/dir/image2.jpg', '/path/to/dir/subdir/image3.jpg']
#     """
#     file_list = []
#     for root, dirs, files in os.walk(root_dir):
#         for file in files:
#             if extension == '' or extension == '*' or file.lower().endswith(extension.lower()):
#                 file_list.append(os.path.join(root, file))
#     file_list.sort(key=lambda x: os.path.getmtime(x))
#     return file_list

# def find_files2(root_dir, pattern='*', recursive=True):
#     """
#     Example:
#     list_of_files = find_files2('c:/temp/', '*.jpg', recursive=True) 
#         returns all jpg files under c:/temp/ in a recursive way. 

#     Parameters
#     ----------
#     root_dir : TYPE
#         DESCRIPTION.
#     pattern : TYPE, optional
#         DESCRIPTION. The default is '*'.

#     Returns
#     -------
#     files : TYPE
#         DESCRIPTION.

#     """
#     if root_dir.count('\\') > root_dir.count('/'):
#         sep = '\\'
#     else:
#         sep = '/'
#     # make sure root_dir is ended with a slash or a backslash                                            
#     if root_dir[-1] != '/' and root_dir[-1] != '\\':
#         root_dir += sep
#     # set glob pathname
#     pathname = root_dir + '**' + sep + pattern   
#     files = glob.glob(pathname, recursive=True)
#     files.sort(key=lambda x: os.path.getmtime(x))
#     return files


class Filelist:
    """
    This class organizes a list of files.
    This class only manages the list, and does not change the actual
    directories or files. 
    
    Its data include:
        filelist
        
    Its functions include:
        
        __init__(listOfFilenames)
        
        clear(): 
            Clears the list, making it an empty list []. 

        extendByList(listOfFilenames):
            Extends the list with the given list.
            For example: 
                cam1_files.extendByList(['c:/f1.jpg', 'c:/f2.jpg'])

        extendByString(pattern)
            Extends the list with files that satisfy given pattern.
            The pattern can be wildcarded (*, ?, **) or c-formatted (%d). 
            For example:
                cam1_files.extendByString('c:/f*.jpg')
                cam1_files.extendByString('c:/f00?5.jpg')
                cam1_files.extendByString('c:/path/**/DSC*.JPG')
                cam1_files.extendByString('c:/path/DSC%05d.JPG, 10, 5') # start, count
                cam1_files.extendByString('c:/path/DSC%05d.JPG, 10, 5, 2') # start, count, step

        sortByTime()
            Sorts the file lists by file time, in ascending order. 

        replaceSubstring(old_dir, new_dir)
            Replaces a part of paths in the list. E.g., 'd:/d1/' --> 'c:/d2/'
            For example: 
                cam1_files.changeDirectoryOfFiles('d:/d1/', 'c:/d2/')   
            Or
                cam1_files.changeDirectoryOfFiles('DSC0', 'IMG0')   

        changeDirectoryOfFiles(new_dir)
            Replaces the old directory of the files to the new one.
            For example:
                cam1_files.changeDirectoryOfFiles('c:/d2/')
            For each file, the entire directory will be replaced. For example, 
                'd:/p1/p2/file1.JPG' will be replaced with 'c:/d2/file1.JPG'
                'd:/p1/file2.JPG' will be replaced with 'c:/d2/file2.JPG'
            If old files are in different directory, this function will destroy
            the directory structure. Think twice before using this function. 

        saveToTxtFile(txt_file)
            Saves the list of files into a text file, one line for each file.
            For example:
                cam1_files.saveToTxtFile('c:/path/cam1_files.txt')

        loadFromTxtFile(txt_file)
            For example:
                cam1_files.loadFromTxtFile('c:/path/cam1_files.txt') 

        nFiles()
            Returns number of files. Equivalent to len(cam1_files.filelist)

        nExistedFiles()
            Returns the number of existed files in the list

        file(i)
            Returns the file, given the index (0 based)
            cam1_files(i) is equal to cam1_files.filelist[i]

        latestExistedFileIndex()
            Returns the latest existed (by file time)

        isImageFile(i)
            returns whether a file is an image that can be read by OpenCV, by 
            actually reading this file and check the data type.

        isImageFilename(i)
            returns whether a file is an image that can be read by OpenCV, by 
            only checking the file name. 

    """
    def __init__(self, listOfFilenames: list=""):
        """
        Example:
        --------
        cam1_files = Filelist(['c:/image1.jpg', 'c:/image2.jpg', 'c:/image3.jpg'])

        """
        self.filelist = []
        if type(listOfFilenames) == list and len(listOfFilenames) > 0:
            self.extendByListOfFilenames(listOfFilenames)

    def clear(self):
        self.filelist = []
        return
            
    def extendByList(self, listOfFilenames):
        """
        Note: This function extends the self.filelist, not replaces it with files.
        If you want to replace it with new files, call function clear() before 
        calling this function.
        """
        if type(listOfFilenames) == list:
            self.filelist.extend(listOfFilenames)
            
    def extendByString(self, string):
        """
        This function extends the file list by adding files that satisfies 
        the criterion defined by the string. 
        Note: This function extends the self.filelist, not replaces it with files.
        If you want to replace it with new files, call function clear() before 
        calling this function.
        It would be better to give full path file name as sometimes you are not
        sure where your current directory is when running this function.
        Here are some examples (assuming cam1_files = Filelist())

        Example 1 (single file)
        cam1_files.extendByString('c:/path/DCIM0001.JPG')

        Example 1 (wildcard of *):
        ---------
        cam1_files.extendByString('c:/DCIM*.JPG')         
        
        Example 2 (wildcard of ?)
        ---------
        cam1_files = Filelist()
        cam1_files.extendByString('c:/DCIM????.JPG')         

        Example 3 (c-format using start and count)
        ---------
        cam1_files = Filelist()
        cam1_files.extendByString('c:/DCIM%04d.JPG, 100, 3')         
        # returns ['c:/DCIM0100.JPG', 'c:/DCIM0101.JPG', 'c:/DCIM0102.JPG']

        Example 4 (c-format using start, end, step) (Notice: end is not included)
        ---------
        cam1_files = Filelist()
        cam1_files.extendByString('c:/DCIM%04d.JPG, 200, 206, 2') 
        # returns ['c:/DCIM0200.JPG', 'c:/DCIM0202.JPG', 'c:/DCIM0204.JPG']
        """
        self.filelist.extend(strToFilelist(string))
        return
        
    def sortByTime(self):
        if len(self.filelist) > 1:
            self.filelist.sort(key=lambda x: os.path.getmtime(x))
        return
    
    def changeDirectoryOfFiles(self, new_dir):
        if len(self.filelist) > 0:
            for i in range(len(self.filelist)):
                thePath = self.filelist[i]
                thePath_dir, thePath_fname = os.path.split(thePath)
                newPath = os.path.join(new_dir, thePath_fname)
                self.filelist[i] = newPath
        return
    
    def replaceSubstring(self, oldSubstr, newSubstr):
        """
        This function is designed for movement of files. For example:
        theFiles.replaceSubstring('d:/mine/test1/', 'e:/yours/test1/')
        will change all files from:
        'd:/mine/test1/cam1/DSCF0001.JPG'
        'd:/mine/test1/cam1/DSCF0002.JPG' ... 
        to 
        'e:/yours/test1/cam1/DSCF0001.JPG'
        'e:/yours/test1/cam1/DSCF0002.JPG' ... 
        
        Parameters
        ----------
        oldSubstr : TYPE
            DESCRIPTION.
        newSubstr : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        if len(self.filelist) > 0:
            for i in range(len(self.filelist)):
                thePath = self.filelist[i]
                newPath = thePath
                if newPath.count('\\') > newPath.count('/'):
                    newPath.replace('/', '\\')
                    oldSubstr.replace('/', '\\')
                    newSubstr.replace('/', '\\')
                else:
                    newPath.replace('\\', '/')
                    oldSubstr.replace('\\', '/')
                    newSubstr.replace('\\', '/')
                newPath.replace(oldSubstr, newSubstr)
                self.filelist[i] = newPath
        return
    
    def saveToTxtFile(self, filename):
        """
        This function saves the file list to a text file in the followin format:
        file1
        file2
        file3...
        For example, if you run theFilelist.saveToTxtFile('c:/t.txt'), 
        the text file c:/t.txt would look like:
        c:\Experiment\Camera1\DSCF0001.JPG
        c:\Experiment\Camera1\DSCF0002.JPG
        c:\Experiment\Camera1\DSCF0003.JPG
        and so on

        Returns
        -------
        None.
        """
        with open(filename, 'w') as file:
            for item in self.filelist:
                file.write("%s\n" % item)
        return
        
    def loadFromTxtFile(self, filename):
        with open(filename, 'r') as file:
            string_list = file.readlines()
        self.filelist = [item.strip() for item in string_list]
        return
    
    def nFiles(self):
        return len(self.filelist)
    
    def nExistedFiles(self):
        count = 0
        for file_name in self.filelist:
            if os.path.exists(file_name):
                count += 1
        return count

    def file(self, i:int):
        return self.filelist[i]
    
    def latestExistedFileIndex(self):
        latest_file = -1
        latest_time = None
        for i in len(self.filelist):
            file_name = self.filelist[i]
            if os.path.exists(file_name):
                file_time = os.path.getmtime(file_name)
                if latest_time is None or file_time > latest_time:
                    latest_file = i
                    latest_time = file_time
        return i
    
    def isImageFile(self, i: int):
        file_name = self.filelist[i]
        img = cv.imread(file_name)
        if type(img) != type(None) and img.size > 0:
            return True
        else:
            return False
    
    def isImageFilename(self, i: int):
        file_name = self.filelist[i]
        extension = os.path.splitext(file_name)[1].lower()
        supportedExt = ['.bmp', '.dib', '.jpeg', '.jpg', '.jpe', '.jp2', '.png',\
                        '.webp', '.pbm', '.pgm', '.ppm', '.pxm', '.pnm', '.sr',\
                        '.ras', '.tif', '.tiff', '.exr', '.hdr', '.pic']
        return extension in supportedExt


#def test_Filelist():
if __name__ == '__main__':
    a = Filelist()
    a.extendByFindFiles2(r'D:\yuansen\ImPro\improMeasure\examples\2022rcwall', 'DSC*.JPG')

