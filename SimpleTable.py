import numpy as np

# SimpleTable is a class that handles a very simplified 2D table with 
# a header row, an index column, and a data matrix.
# The header row is a list of strings, which variable is table_header.
# The index column is a list of strings, which variable is table_index.
# The data matrix is a 2D numpy array, which variable is table_data.
# This class provides the following functionalities:
# 1. Generates a (large) string that represents the table in csv format 
#    (or user defined delimiter), including the header and index.
# 2. Saves the table to a csv file (or user defined delimiter), 
#    including the header and index.
# 3. Loads the table from a csv file (or user defined delimiter).

class SimpleTable:
    # Constructor
    def __init__(self, table_header=None, table_index=None, table_data=None):
        if table_header is None:
            table_header = []
        if table_index is None:
            table_index = []
        if table_data is None:
            table_data = np.array([[]])
        self.table_header = table_header
        self.table_index = table_index
        self.table_data = table_data

    # This function returns the number of rows in the table.
    def get_num_rows(self):
        return self.table_data.shape[0]

    # This function returns the number of columns in the table.
    def get_num_columns(self):
        return self.table_data.shape[1]

    # This function returns the header row of the table.
    def get_header(self):
        return self.table_header

    # This function returns the index column of the table.
    def get_index(self):
        return self.table_index

    # This function returns the data matrix of the table.
    def get_data(self):
        return self.table_data

    # This function sets the header row of the table.
    def set_header(self, header):
        self.table_header = header

    # This function sets the index column of the table.
    def set_index(self, index):
        self.table_index = index

    # This function sets the data matrix of the table.
    def set_data(self, data):
        self.table_data = data

    # This function generates a string that represents the table in csv format.
    # The delimiter is a comma by default.
    def to_csv(self, delimiter=','):
        # generate the header
        header_str = delimiter.join(self.table_header)
        # generate the index
        index_str = delimiter + delimiter.join(self.table_index) + '\n'
        # generate the data
        data_str = ''
        #    generate the header part of the data_str
        #    for example, if the header_str is 'A,B,C', then the header part of the data_str is ',A,B,C'
        #    this part is used to align the data with the header
        #    If the length of the header is the width of table, the first delimiter is added (meaning the first word is empty).
        #    if the length of the header is the width of table plus one,  the delimiter is not added (meaning the first word is in the header).
        if self.table_data.shape[1] == len(self.table_header):
            data_str += delimiter + header_str + '\n'
        else:
            data_str += header_str + '\n'
        #    add the index and data
        for i in range (self.table_data.shape[0]):
            # the index is the first element of each row
            data_str += self.table_index[i] + delimiter + delimiter.join(map(str, self.table_data[i])) + '\n'
        return data_str

    # This function saves the table to a csv file.
    # The delimiter is a comma by default.
    def save_to_csv(self, filename, delimiter=','):
        with open(filename, 'w') as f:
            f.write(self.to_csv(delimiter))

    # This function loads the table from a csv file
    # The delimiter is a comma by default.
    def load_from_csv(self, filename, delimiter=','):
        with open(filename, 'r') as f:
            lines = f.readlines()
            # read the header
            self.table_header = lines[0].strip().split(delimiter)
            # read the index and data
            self.table_index = []
            self.table_data = []
            for line in lines[1:]:
                parts = line.strip().split(delimiter)
                self.table_index.append(parts[0])
                self.table_data.append(list(map(float, parts[1:])))
            self.table_data = np.array(self.table_data)

# a unit test of SimpleTable
if __name__ == '__main__':
    # create a SimpleTable object
    table = SimpleTable()
    # set the header
    table.set_header(['A', 'B', 'C'])
    # set the index
    table.set_index(['Object 1', 'Object 2', 'Object 3'])
    # set the data
    table.set_data(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
    # print the table
    print(table.to_csv())
    # save the table to a csv file
    # You will get the file that looks like this:
    # ,A,B,C
    # Object 1,1,2,3
    # Object 2,4,5,6
    # Object 3,7,8,9
    # which you can open it with a spreadsheet program like Excel.
    table.save_to_csv('tests/test_SimpleTable.csv')
    # load the table from the csv file
    table.load_from_csv('tests/test_SimpleTable.csv')
    # save the table to a csv file
    # You will get the file that should be exactly the same as the previous one.
    table.save_to_csv('tests/test_SimpleTable2.csv')
    # print the table
    print(table.to_csv())



