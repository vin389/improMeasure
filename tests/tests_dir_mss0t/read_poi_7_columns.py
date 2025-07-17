import csv
def read_poi_7_columns(file_path):
    """
    Read a CSV file with 7 columns: poi_name, xi, yi, x0, y0, w, h.
    Returns a list of tuples (poi_name, xi, yi, x0, y0, w, h).
    """
    # check if file_path is a valid string
    if not isinstance(file_path, str) or not file_path.strip():
        print("# Error: Invalid file path provided: %s." % file_path)
        return None
    # initialize lists to store poi_names, Xi, and Xir
    poi_names = []
    Xi = []
    Xir = []
    # try to read from the file 
    try:
        with open(file_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)  # Skip header line
            for row in reader:
                if len(row) == 7:
                    poi_name = row[0]
                    xi = float(row[1])
                    yi = float(row[2])
                    x0 = float(row[3])
                    y0 = float(row[4])
                    w = float(row[5])
                    h = float(row[6])
                    poi_names.append(poi_name)
                    Xi.append((xi, yi))
                    Xir.append((x0, y0, w, h))
        return poi_names, Xi, Xir
    except:
        print("# Error: Failed to read data from the file %s." % file_path)
        return None
    
# Use tkinter to create a GUI unit test
# The first of the tk window is a button that allows the user to select a file
# The second is a text area that shows the content of the file
if __name__ == "__main__":
    import tkinter as tk
    from tkinter import filedialog, messagebox

    def select_file():
        file_path = filedialog.askopenfilename(
            title="Select a CSV file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if file_path:
            result = read_poi_7_columns(file_path)
            if result is not None:
                poi_names, Xi, Xir = result
                num_pois = len(poi_names)
                str_print = "# Number of POIs: %d\n" % num_pois
                for i in range(num_pois):
                    str_print += "# POI %d: %s, Xi: (%f, %f), Xir: (%d, %d, %d, %d)\n" % (
                        i + 1, poi_names[i], Xi[i][0], Xi[i][1], Xir[i][0], Xir[i][1], Xir[i][2], Xir[i][3])
                output_text.set(str_print)
            else:
                output_text.set("User cancelled the dialog.")

    root = tk.Tk()
    root.title("Read POI 7 Columns")

    select_button = tk.Button(root, text="Select a CSV file that contains POI data with 7 columns (poi_name, xi, yi, x0, y0, w, h)", command=select_file)
    select_button.pack(pady=10)

    output_text = tk.StringVar()
    output_label = tk.Label(root, textvariable=output_text, justify=tk.LEFT)
    output_label.pack(pady=10)

    root.mainloop()