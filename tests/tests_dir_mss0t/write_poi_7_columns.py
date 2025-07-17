import csv
def write_poi_7_columns(poi_names, Xi, Xir, file_path):
    """
    Write a CSV file with 7 columns: poi_name, xi, yi, x0, y0, w, h.
    Each POI is represented by its name, coordinates (xi, yi), and rectangle (x0, y0, w, h).
    The input lists should be of the same length, and each POI corresponds to one row in the CSV file.
    If the input lists are empty, the function will create an empty file with only the header.
    Args:
    - poi_names: List of POI names (strings).
    - Xi: List of tuples (xi, yi) representing the coordinates of each POI.
    - Xir: List of tuples (x0, y0, w, h) representing the rectangle for each POI.
    """
    # check if file_path is a valid string
    if not isinstance(file_path, str) or not file_path.strip():
        print("# Error: Invalid file path provided: %s." % file_path)
        return False
    
    # check if the input lists are of the same length
    if len(poi_names) != len(Xi) or len(poi_names) != len(Xir):
        print("# Error: Input lists must have the same length. (poi_names: %d, Xi: %d, Xir: %d)" % (len(poi_names), len(Xi), len(Xir)))
        return False
    
    # try to write to the file 
    try:
        # open the file in write mode
        with open(file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # write header
            writer.writerow(['poi_name', 'xi', 'yi', 'x0', 'y0', 'w', 'h'])
            # write data rows
            for i in range(len(poi_names)):
                writer.writerow([poi_names[i], Xi[i][0], Xi[i][1], Xir[i][0], Xir[i][1], Xir[i][2], Xir[i][3]])
        return True
    except Exception as e:
        print("# Error: Failed to write data to the file %s. Exception: %s" % (file_path, str(e)))
        return False

# Example usage
# Create a tk gui to test the function
# Use inputdlg3 to get poi data from user
if __name__ == "__main__": 
    import tkinter as tk
    from tkinter import filedialog, messagebox
    from inputdlg3 import inputdlg3
    title = "Enter POI data and write to file"
    prompts = [
        "POI names",
        "POI coordinates",
        "POI rectangles",
        "Output file"
    ]
    datatypes = [
        'string',  # POI names
        'array -1 2',  # POI coordinates
        'array -1 4',  # POI rectangles
        'filew'   # Output file path
    ]
    initvalues = [
        "POI1,POI2,POI3",  # Example POI names
        "100.0,200.0 , 300.0,400.0 , 500.0,600.0",  # Example POI coordinates (xi,yi)
        "50,50,100,100 , 150,150,200,200 , 250,250,300,300",  # Example POI rectangles (x0,y0,w,h)
        ""  # Output file path (to be selected by user)
    ]
    tooltips = [
        "POI names (comma separated):",
        "POI coordinates xi_1, yi_1, xi_2, yi_2, ...  (comma separated):",
        "POI rectangles x0_1,y0_1,w_1,h_1, x0_2,y0_2,w_2,h_2, ... (comma separated):",
        "Output file path:"
    ]
    result = inputdlg3(title=title, 
                       prompts=prompts, 
                       initvalues=initvalues, 
                       datatypes=datatypes, 
                       tooltips=tooltips)
    if result is not None:
        poi_names = result[0].split(',')
        Xi = result[1]
        Xir = result[2].astype(int)
        output_file = result[3]
        
        # Write to file
        if write_poi_7_columns(poi_names, Xi, Xir, output_file):
            messagebox.showinfo("Success", f"POI data written to {output_file}")
        else:
            messagebox.showerror("Error", "Failed to write POI data to file.")
