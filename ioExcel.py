import os
import pandas as pd

def writeDataFrameToExcel(df, xlsFile, sheetName):
    """
    This function writes a pandas DataFrame (df) to a sheet 
    (sheet_name) of an Excel file (xlsFile). If the Excel file
    has been existed, load it befoe writing to keep existing 
    sheets intact. 
    """
    if os.path.exists(xlsFile) == False:
        # if new file
        df.to_excel(xlsFile, sheet_name=sheetName)
    else:
        with pd.ExcelWriter(xlsFile, engine='openpyxl', mode='a',
             if_sheet_exists='overlay') as writer:
            df.to_excel(writer, sheet_name=sheetName)
