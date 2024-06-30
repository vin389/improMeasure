import numpy as np

# Sample 2D array
data = np.array([[1, 2, 3], [4, 5, 6]]) * np.pi 

# Header row
headerori = "column1,column2,column3".split(',')
header = headerori[:]
header[0] = '# ' + header[0]

# Combine header and data using vstack
data_with_header = np.vstack((header, data))

# Save the data with header using np.savetxt with delimiter set as ',' and fmt='%s' for string type
np.savetxt("data_with_header.txt", data_with_header, delimiter=",", fmt="%s")

print("Data saved to data_with_header.txt")
print(headerori)
