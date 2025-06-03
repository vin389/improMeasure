# TabularData is a class that organizes data in a tabular format.
# It loads and saves tabular data from/to a file.
# The first row of the data is a header row, which contains the names of the columns.
# The first column of the data is the name of the row.
# After importing the data from the file, the data is stored in a numpy array.
# The function self.get_row('row_name') returns the row with the given name.

import numpy as np
import re
import cv2
import csv
import os

class TabularData:
    def __init__(self):
        pass

    def create_demo_camera_parameters(self):
        # 
        self.header = ['parameter_name','cam1', 'cam2', 'cam3', 'cam4']
        self.row_names = ['image_width', 'image_height', 'rvec_x', 'rvec_y', 'rvec_z', 'tvec_x', 'tvec_y', 'tvec_z', 'cmat_11', 'cmat_12', 'cmat_13', 'cmat_21', 'cmat_22', 'cmat_23', 'cmat_31', 'cmat_32', 'cmat_33', 'dvec_k1', 'dvec_k2', 'dvec_p1', 'dvec_p2', 'dvec_k3', 'dvec_k4', 'dvec_k5', 'dvec_k6', 'dvec_s1', 'dvec_s2', 'dvec_s3', 'dvec_s4', 'dvec_taux', 'dvec_tauy']
        self.data = np.data = np.array(
           [[ 1.920000e+03,  1.920000e+03,  1.920000e+03,  1.920000e+03],
            [ 1.080000e+03,  1.080000e+03,  1.080000e+03,  1.080000e+03],
            [ 1.209200e+00,  0.000000e+00,  1.209200e+00,  1.570796e+00],
            [ 1.209200e+00,  2.221441e+00, -1.209200e+00,  0.000000e+00],
            [-1.209200e+00, -2.221441e+00,  1.209200e+00,  0.000000e+00],
            [ 0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00],
            [ 0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00],
            [ 3.000000e+00,  3.000000e+00,  3.000000e+00,  3.000000e+00],
            [ 9.600000e+02,  9.600000e+02,  9.600000e+02,  9.600000e+02],
            [ 0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00],
            [ 9.595000e+02,  9.595000e+02,  9.595000e+02,  9.595000e+02],
            [ 0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00],
            [ 9.600000e+02,  9.600000e+02,  9.600000e+02,  9.600000e+02],
            [ 5.395000e+02,  5.395000e+02,  5.395000e+02,  5.395000e+02],
            [ 0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00],
            [ 0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00],
            [ 1.000000e+00,  1.000000e+00,  1.000000e+00,  1.000000e+00],
            [ 0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00],
            [ 0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00],
            [ 0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00],
            [ 0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00],
            [ 0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00],
            [ 0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00],
            [ 0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00],
            [ 0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00],
            [ 0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00],
            [ 0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00],
            [ 0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00],
            [ 0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00],
            [ 0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00],
            [ 0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00]])
        pass        

    def load(self, filename=None):
        # Load the tabular data from a CSV file.
        # if the filename is not provided, it calls uigetfile() to get the filename from the user.
        if filename is None:
            from uigetfile import uigetfile
            f_uigetfile = uigetfile()
            filename = os.path.join(f_uigetfile[0], f_uigetfile[1])
            if len(f_uigetfile[0]) == 0 and len(f_uigetfile[1]) == 0:
                print("# User cancelled file dialog. No file selected.")
                return
        # if filename does not exist, print an error message and return
        if not os.path.isfile(filename):
            print(f"# Error: The file {filename} does not exist.")
            return
        # try to open the file and read the data
        # Read the header row of the CSV file, which contains the names of the columns, 
        # and stores it in a list of strings, self.header
        # The first column are names of each row, which are then stored in self.row_names
        # The self.row_names is a list of strings, where each string is the name of a row.
        # The self.row_names starts from the 2nd row of the first column of the CSV file.
        # The rest of the data are suppposed to be numeric values, which are stored 
        # in a numpy array self.data, which dtype is np.float64
        try:
            # read the csv file by np.genfromtxt
            np_data = np.genfromtxt(filename, delimiter=',', dtype=None, skip_header=0, encoding=None, comments='#')
            # self.header is a list of strings, which is the first row of the np_data
            self.header = np_data[0].astype(str).tolist()
            # self.row_names is a list of strings, which is the first column of the np_data
            self.row_names = np_data[1:, 0].astype(str).tolist()
            # self.data is the rest of the np_data, which is a 2D numpy array
            self.data = np_data[1:, 1:].astype(np.float64)
        except:
            print(f"# Error: Cannot read the file {filename}. Please check the file format.")
            print("#  The file should be a CSV file.")
            print("#  The first row should be the header, which contains the names of the columns.")
            print("#  The first column should be the names of the rows.")
            print("#  The rest of the data should be numeric values.")
            print("#  For example (ignore the hash because hash is comment.):")
            print("#  Object_name, property_1, property_2, property_3")
            print("#  name1, 1.0, 2.0, 3.0")
            print("#  name2, 4.0, 5.0, 6.0")
            print("#  name3, 7.0, 8.0, 9.0")
            return
        
    def save(self, filename=None):
        # save the tabular data to a CSV file.
        # if the filename is not provided, it calls uiputfile() to get the filename from the user.
        from uiputfile import uiputfile
        if filename is None:
            f_uiputfile = uiputfile()
            filename = os.path.join(f_uiputfile[0], f_uiputfile[1])
            if len(f_uiputfile[0]) == 0 and len(f_uiputfile[1]) == 0:
                print("# User cancelled file dialog. No file selected.")
                return
        # write the self.header, self.row_names, and self.data to the file
        # for example, if the header is ['Object_name', 'property_1', 'property_2', 'property_3']
        # and the self.row_names is ['name1', 'name2', 'name3'] and the data is a 2D numpy array
        # [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        # Write the file in the following format:
        # Object_name, property_1, property_2, property_3
        # name1, 1.0, 2.0, 3.0
        # name2, 4.0, 5.0, 6.0
        # name3, 7.0, 8.0, 9.0
        try:
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                # write the header
                writer.writerow(self.header)
                # write the data
                for i, row_name in enumerate(self.row_names):
                    writer.writerow([row_name] + self.data[i].tolist())
        except Exception as e:
            print(f"# Error: Cannot write to the file {filename}. Please check the file format.")
            print(f"#  Exception: {e}")
            return
        
    def get_row(self, row_name):
        # Get the row with the given name.
        # If the row name is not found, it returns None.
        if row_name in self.row_names:
            index = self.row_names.index(row_name)
            return self.data[index]
        else:
            print(f"# Error: Row name '{row_name}' not found.")
            return None
        
    def get_column(self, column_name):
        # Get the column with the given name.
        # If the column name is not found, it returns None.
        if column_name in self.header:
            index = self.header.index(column_name) - 1
            return self.data[:, index]
        else:
            print(f"# Error: Column name '{column_name}' not found.")
            return None
        
    def get_element(self, row_name, column_name):
        # Get the element at the given row and column.
        # if the row_name is not found, try to find if row_

        # If the row name or column name is not found, it returns None.
        if row_name in self.row_names and column_name in self.header:
            row_index = self.row_names.index(row_name)
            column_index = self.header.index(column_name) - 1
            return self.data[row_index, column_index]
        else:
            # check if the row_


            print(f"# Error: Row name '{row_name}' or column name '{column_name}' not found.")
            return None 
        
    def add_row(self, row_name, row_data):
        # Add a new row to the data.
        # If the row name already exists, it updates the existing row.
        # row_data can be a 1D numpy array or a list with the same number of columns as the header
        # or a scalar value 
        # Example: 
        #   this_object.add_row('new_row', [1.0, 2.0, 3.0, 4.0])
        if row_name in self.row_names:
            index = self.row_names.index(row_name)
            self.data[index] = row_data
        else:
            # if row_data is a scalar, convert it to a 1D numpy array
            # of the same length as the header 
            if isinstance(row_data, (int, float)):
                row_data = np.array([row_data] * (len(self.header) - 1), dtype=np.float64)
            # if row_data length is less than the number of columns in the header, 
            # expand it with zeros
            elif len(row_data) < len(self.header) - 1:
                row_data = np.array(row_data + [0] * (len(self.header) - 1 - len(row_data)), dtype=np.float64)
            # Check if row_data has the same number of columns as the header
            if len(row_data) != len(self.header) - 1:
                print(f"# Error: Row data for '{row_name}' must have {len(self.header) - 1} elements.")
                return
            # Add the new row
            self.row_names.append(row_name)
            self.data = np.vstack([self.data, row_data])

    def add_column(self, column_name, column_data):
        # Add a new column to the data.
        # If the column name already exists, it updates the existing column.
        # column_data can be a 1D numpy array or a list with the same number of rows as the data
        # Example: 
        #   cameras.add_column('new_col', [1.0, 2.0, 3.0, 4.0])
        #   cameras.add_column('cam5', np.array([1920, 1080, 1.2092, 0.0, 1.2092, 1.5708, 0.0, 3.0, 
        #       960.0, 0.0, 960.0, 0.0, 960.0, 539.5, 0.0, 1, 0., 0., 0., 0., 0., 0.])) 
        if column_name in self.header:
            index = self.header.index(column_name)
            self.data[:, index - 1] = column_data
        else:
            # if column_data is a scalar, convert it to a 1D numpy array
            # of the same length as the number of rows in the data
            if isinstance(column_data, (int, float)):
                column_data = np.array([column_data] * len(self.row_names), dtype=np.float64)
            # Check if column_data has the same number of rows as the data
            if len(column_data) != len(self.row_names):
                print(f"# Error: Column data for '{column_name}' must have {len(self.row_names)} elements.")
                return
            # Add the new column
            self.header.append(column_name)
            new_column = np.array(column_data).reshape(-1, 1)
            self.data = np.hstack([self.data, new_column])

    def remove_row(self, row_name):
        # Remove a row from the data.
        # If the row name is not found, it does nothing.
        if row_name in self.row_names:
            index = self.row_names.index(row_name)
            self.row_names.pop(index)
            self.data = np.delete(self.data, index, axis=0)
        else:
            print(f"# Error: Row name '{row_name}' not found.")

    def remove_column(self, column_name):
        # Remove a column from the data.
        # If the column name is not found, it does nothing.
        if column_name in self.header:
            index = self.header.index(column_name)
            self.header.pop(index)
            self.data = np.delete(self.data, index - 1, axis=1)

if __name__ == "__main__":
    # Example usage
    t = TabularData()
    t.create_demo_camera_parameters()
    print("Header:", t.header)
    print("Row names:", t.row_names)
    print("Data:\n", t.data)
    
    # Save to a file
    t.save("d:/temp/camera_parameters.csv")
    
    # Load from a file
    new_t = TabularData()
    new_t.load("d:/temp/camera_parameters.csv")
    print("Loaded Header:", new_t.header)
    print("Loaded Row names:", new_t.row_names)
    print("Loaded Data:\n", new_t.data)        