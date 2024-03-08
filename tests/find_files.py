import os 

# Specify the directory to search and the file extension to match
# directory = r'D:\ExpDataSamples\tmp\files'
# extension = '.JPG'


# def find_files(directory, extension):
#     for root, dirs, files in os.walk(directory):
#         for file in files:
#             if file.endswith(extension):
#                 yield os.path.join(root, file)

def find_files(directory, extension):
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(extension):
                file_list.append(os.path.join(root, file))
    return file_list

#files = find_files(directory, extension)
