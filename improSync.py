import PIL
from PIL import Image
from PIL.ExifTags import TAGS
import datetime
import numpy as np
import glob
import os
import time

def savePhotoTimeTable(imgFiles, csvFile: str, ):
    # converts to list of strings (imgFiles can have wildcard such as * or ? )
    if type(imgFiles) == str:
        fileList = glob.glob(imgFiles)
    if type(imgFiles) == list:
        fileList = imgFiles
    if type(fileList) != list: 
        return -1 # returns error
    nFiles = len(fileList)
    # create list of image exif DateTime
    dtimeList = []
    for i in range(nFiles):
        img = Image.open(fileList[i])
        exif = img.getexif()
        exifDateTime = exif.get(306)  # tag id 306 is DateTime
        if type(exifDateTime) == str:
            dtimeList.append(exifDateTime)
        else:
            dtimeList.append('')
    # Save data frame to csv file 
    try:
        file = open(csvFile, 'w')
        file.write('Path, File, Exif DateTime, Year, Month, Day, Hour, Minute, Sec'
                   ', Total seconds since 1970-01-01\n')
        exifFormat = '%Y:%m:%d %H:%M:%S'
        for i in range(nFiles):
            thePathFile = os.path.split(fileList[0])
            thePath = thePathFile[0]
            theFile = thePathFile[1]
            theDt = datetime.datetime.strptime(dtimeList[i], exifFormat)
            # parse date-time string to integers year/month/day/hour/minute/sec
            total_sec_1970 = (theDt - datetime.datetime(1970, 1, 1)).total_seconds()
            file.write("%s, %s, %s, %s, %d\n" % \
                       (thePath, theFile, dtimeList[i], 
                        theDt.strftime('%Y, %m, %d, %H, %M, %S'), 
                        total_sec_1970))
        file.close()
    except:
        print("# Error: savePhotoTimeTable() failed to write to file.")
        return -2
    
        




def getExifDateTime(fname):
    img = Image.open(fname)
    exif = img.getexif()
    # tag_id 306: DateTime
    # tag_id 36867: DateTimeOriginal
    # tag_id 36868: DateTimeDigitized
    exifDateTime = exif.get(306)
    img.close()
    return exifDateTime

# fname = r'D:/ExpDataSamples/20220500_Brb/brb1/brb1_cam1_northGusset/IMG_4234.JPG'
# img = Image.open(fname)
# theExif = img.getexif()
# for tag_id in theExif:
#     # get the tag name, instead of human unreadable tag id
#     tag = TAGS.get(tag_id, tag_id)
#     data = theExif.get(tag_id)
#     # decode bytes 
#     if isinstance(data, bytes):
#         data = data.decode()
#     print(f"{tag:25}: {data}")
# img.close()

# for i in range(999999):
#     tags = TAGS.get(i,i)
#     if type(tags)==type('str'):
#         if tags.find('Time') >= 0 or tags.find('time') >= 0 or\
#             tags.find('Date') >= 0 or tags.find('date') >= 0:
#             print('%d:%s', i, tags)
        
        