import os
import glob
import numpy as np
import cv2 as cv

def ifunc_p2v(pic_files, ):
    """
    This function converts pictures to a single video. 
    User specifies:
        files: 

    Returns
    -------
    None.

    """



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
