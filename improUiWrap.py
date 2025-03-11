from math import * 
import copy
import numpy as np
import cv2
import os, sys, glob, time, re
import tkinter as tk
from npFromStr import npFromStr
from str2argDict import *
from ArgFloat import ArgFloat

def improUiWrap(inArgList, outArgList, theFunction, winTitle='GUI for function'):
    # Create a tk window that contains a grid of widgets
    # The widgets are created based on the arguments
    # The arguments are given as a list of ArgXXX objects, such as ArgFloat, ArgInt, ArgImage, etc.

    # Create a tk window
    root = tk.Tk()
    root.title(winTitle)

    # Create a list of widgets
    widgets = {}
    # Create a label on the top of the windown
    #  header of 'label' of input argument
    lb_type = tk.Label(root, text='type')
    lb_type.grid(row=0, column=0)
    #  header of 'name' of input argument
    lb_name = tk.Label(root, text='name')
    lb_name.grid(row=0, column=1)
    #  header of 'description' of input argument
    lb_desc = tk.Label(root, text='description')
    lb_desc.grid(row=0, column=2)
    #  header of 'entry' of input argument
    lb_entry = tk.Label(root, text='entry')
    lb_entry.grid(row=0, column=3)
    #  header of 'check' of input argument
    lb_check = tk.Label(root, text='check')
    lb_check.grid(row=0, column=4)
    #  header of 'load' of input argument
    lb_load = tk.Label(root, text='file dialog')
    lb_load.grid(row=0, column=5)

    ######################################################
    # Input arguments
    ######################################################
    # for each argDict in argDicts
    for i in range(len(inArgList)):
        #############################################
        # create a label of argument type 
        #   label type
        lb_type = tk.Label(root, text=inArgList[i].dtype())
        lb_type.grid(row=i+1, column=0)
        #  the pointer of the widget is at widgets['inArg_0_type'] for the first argument 
        widgetKey = 'inArg_%d_type' % i
        widgets[widgetKey] = lb_type 
        #############################################
        # create a label of argument name 
        #   label name
        lb_name = tk.Label(root, text=inArgList[i].name)
        lb_name.grid(row=i+1, column=1)
        #  the pointer of the widget is at widgets['inArg_0_name'] for the first argument 
        widgetKey = 'inArg_%d_name' % i
        widgets[widgetKey] = lb_name
        #############################################
        # create a button which text is 'description' with event handler
        #   label description
        # if the button is clicked, it pops up a message box that shows the description of this argument
        bt_desc = tk.Button(root, text='Desc.')
        bt_desc.grid(row=i+1, column=2)
        # add event handler to bt_desc
        def general_show_desc(theArgObject):
            # popup a message box that shows the description
            from tkinter import messagebox
            messagebox.showinfo('Description', theArgObject.funcDesc())
        bt_desc.configure(command=lambda theArgObject=inArgList[i]: general_show_desc(theArgObject))
        #  the pointer of the widget is at widgets['inArg_0_desc'] for the first argument 
        widgetKey = 'inArg_%d_desc' % i
        widgets[widgetKey] = bt_desc 
        #############################################
        # create an entry
        ed_entry = tk.Entry(root, width=60)
        ed_entry.grid(row=i+1, column=3)
        # if 'default' exists in the key of argDict
        if inArgList[i].default != None:
            # set the default value to the entry
            ed_entry.delete(0, tk.END)
            ed_entry.insert(0, inArgList[i].default)
        #  the pointer of the widget is at widgets['inArg_0_entry'] for the first argument 
        widgetKey = 'inArg_%d_entry' % i
        widgets[widgetKey] = ed_entry 
        #############################################
        # create a button with text 'check'
        bt_check = tk.Button(root, text='Check')
        bt_check.grid(row=i+1, column=4)
        # add event handler to bt_check
        def inArg_general_check(idx):
            # get the entry from the widgets
            theEntry = widgets['inArg_%d_entry' % idx]
            # get the value from the entry
            (check_result, check_message) = inArgList[idx].check(theEntry.get())
            # pop up a message box to show the result
            if len(check_message) > 0:
                from tkinter import messagebox
                tk.messagebox.showinfo('Check', check_message)
        bt_check.configure(command=lambda idx=i: inArg_general_check(idx))
        #  the pointer of the widget is at widgets['inArg_0_check'] for the first argument 
        widgetKey = 'inArg_%d_check' % i
        widgets[widgetKey] = bt_check 
        #############################################
        # create a button with text 'browse...' if there are load() and save()
        #   if inArgList[i] has functions load() and save()
        if hasattr(inArgList[i], 'load') and callable(inArgList[i].load) and hasattr(inArgList[i], 'save') and callable(inArgList[i].save):
            bt_browser = tk.Button(root, text='browse...')
            bt_browser.grid(row=i+1, column=5)
            # add event handler to bt_browser for user to pick a file and display the file path in the entry
            def inArg_general_browse(idx):
                # popup a file dialog to get the file path
                from tkinter import filedialog
                root.withdraw()
                file_path = filedialog.askopenfilename(title='Select a file for input argument %s' % inArgList[idx].name)
                root.deiconify()
                # if the file path is not empty
                if len(file_path) > 0:
                    # get the entry from the widgets
                    theEntry = widgets['inArg_%d_entry' % idx]
                    # set the value to the entry
                    theEntry.delete(0, tk.END)
                    theEntry.insert(0, file_path)
            bt_browser.configure(command=lambda idx=i: inArg_general_browse(idx))
            #  the pointer of the widget is at widgets['inArg_0_load'] for the first argument
            widgetKey = 'inArg_%d_browse' % i
            widgets[widgetKey] = bt_browser

    ######################################################
    # create a button with text 'Run'
    ######################################################
    bt_run = tk.Button(root, text='Run')
    bt_run.grid(row=len(inArgList)+1, column=0)
    # add event handler to bt_run
    def run_the_function():
        # get the value from the entry of each input argument
        functionArguments = []
        # for each inArg in inArgList
        for i in range(len(inArgList)):
            theEntry = widgets['inArg_%d_entry' % i]
            inArgList[i].set(theEntry.get())
            completeData = copy.deepcopy(inArgList[i].get())
            functionArguments.append(completeData)
        # run the function with Python argument unpacking syntax (*)
        funcResults = theFunction(*functionArguments)
        # get the returned values from the function
        if type(funcResults) != tuple and type(funcResults) != list:
            funcResults = (funcResults,)
        # put funcResults to the output arguments
        for i in range(len(outArgList)):
            # get the returned value from the function and display it in the entry
            outArgList[i].set(funcResults[i])
            the_string = outArgList[i].toString()
            # if the argument has save function, and the text entry ends with '.txt', '.csv', '.jpg', '.bmp', '.tif', '.png'
            # then save the value to the file
            theEntryText = widgets['outArg_%d_entry' % i].get()
            if hasattr(outArgList[i], 'save') and callable(outArgList[i].save) and \
                 theEntryText.endswith(('.txt', '.csv', '.npy', '.jpg', '.bmp', '.tif', '.png')):
                outArgList[i].save(theEntryText)
            else:
                theEntry = widgets['outArg_%d_entry' % i]
                theEntry.delete(0, tk.END)
                theEntry.insert(0, the_string)
            # # if the string (that represents the value) is not empty, set it to the entry
            # # however, if the string is empty, check if this argument needs to be saved to a file
            # if len(the_string) > 0:
            #     theEntry = widgets['outArg_%d_entry' % i]
            #     theEntry.delete(0, tk.END)
            #     theEntry.insert(0, the_string)
            # else:
            #     # if the string is empty, check if this argument needs to be saved to a file
            #     if hasattr(outArgList[i], 'save') and callable(outArgList[i].save):
            #         # get the entry from the widgets
            #         theEntry = widgets['outArg_%d_entry' % i]
            #         # get the value from the entry
            #         file_path = theEntry.get()
            #         # if the file path is not empty
            #         if len(file_path) > 0:
            #             # save the value to the file
            #             outArgList[i].save(file_path)

    bt_run.configure(command=run_the_function)
    widgets['run'] = bt_run

    ######################################################
    # Output arguments
    ######################################################
    # for each outArg in outArgList
    for i in range(len(outArgList)):
        ######################################################
        # create a label of argument type 
        #   label type
        lb_type = tk.Label(root, text=outArgList[i].dtype())
        lb_type.grid(row=i+len(inArgList)+3, column=0)
        #  the pointer of the widget is at widgets['outArg_0_type'] for the first argument 
        widgetKey = 'outArg_%d_type' % i
        widgets[widgetKey] = lb_type 
        ######################################################
        # create a label of argument name 
        #   label name
        lb_name = tk.Label(root, text=outArgList[i].name)
        lb_name.grid(row=i+len(inArgList)+3, column=1)
        #  the pointer of the widget is at widgets['outArg_0_type'] for the first argument 
        widgetKey = 'outArg_%d_name' % i
        widgets[widgetKey] = lb_name 
        ######################################################
        # create a Button of argument description 
        bt_desc = tk.Button(root, text='Desc.')
        bt_desc.grid(row=i+len(inArgList)+3, column=2)
        # add event handler to bt_desc
        bt_desc.configure(command=lambda theArgObject=outArgList[i]: general_show_desc(theArgObject))
        #  the pointer of the widget is at widgets['outArg_0_desc'] for the first argument 
        widgetKey = 'outArg_%d_desc' % i
        widgets[widgetKey] = bt_desc 
        ######################################################
        # create an entry
        ed_entry = tk.Entry(root, width=60)
        ed_entry.grid(row=i+len(inArgList)+3, column=3)
        if outArgList[i].default != None:
            # set the default value to the entry
            ed_entry.delete(0, tk.END)
            ed_entry.insert(0, outArgList[i].default)
        #  the pointer of the widget is at widgets['outArg_0_entry'] for the first argument 
        widgetKey = 'outArg_%d_entry' % i
        widgets[widgetKey] = ed_entry 
        ######################################################
        # create a button with text 'check'
        bt_check = tk.Button(root, text='Check')
        bt_check.grid(row=i+len(inArgList)+3, column=4)
        # add event handler to bt_check
        def outArg_general_check(idx):
            # directly check the self.value
            (check_result, check_message) = outArgList[idx].check()
            # pop up a message box to show the result
            if len(check_message) > 0:
                from tkinter import messagebox
                tk.messagebox.showinfo('Check', check_message)
        bt_check.configure(command=lambda idx=i: outArg_general_check(idx))
        #  the pointer of the widget is at widgets['outArg_0_check'] for the first argument 
        widgetKey = 'outArg_%d_check' % i
        widgets[widgetKey] = bt_check 
        #############################################
        # create a button with text 'browse...' if there are load() and save()
        #   if outArgList[i] has functions load() and save()
        if hasattr(outArgList[i], 'load') and callable(outArgList[i].load) and hasattr(outArgList[i], 'save') and callable(outArgList[i].save):
            bt_browser = tk.Button(root, text='browse...')
            bt_browser.grid(row=i+len(inArgList)+3, column=5)
            # add event handler to bt_browser for user to pick a file and display the file path in the entry
            def outArg_general_browse(idx):
                # popup a file dialog to get the file path
                from tkinter import filedialog
                root.withdraw()
                file_path = filedialog.asksaveasfilename(title='Select a file for output argument %s' % outArgList[idx].name)
                root.deiconify()
                # if the file path is not empty
                if len(file_path) > 0:
                    # get the entry from the widgets
                    theEntry = widgets['outArg_%d_entry' % idx]
                    # set the value to the entry
                    theEntry.delete(0, tk.END)
                    theEntry.insert(0, file_path)
            bt_browser.configure(command=lambda idx=i: outArg_general_browse(idx))
            #  the pointer of the widget is at widgets['inArg_0_load'] for the first argument
            widgetKey = 'outArg_%d_browse' % i
            widgets[widgetKey] = bt_browser



    root.mainloop()

def lawOfCosine(a, b, angle):
    c = sqrt(a**2 + b**2 - 2*a*b*cos(angle))
    area = 0.5 * a * b * sin(angle)
    return (c, area)

# a function that calculates the area of a triangle given three x-y coordinates of the vertices
def areaOfTriangle(x1, y1, x2, y2, x3, y3):
    area = 0.5 * abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))
    return area


if __name__ == '__main__':


    # example 1: triangle side length and angle given two sides and the angle between them
    arg1 = ArgFloat()
    arg1.fromString('--type float --name edge1 --desc The length of the first side of the triangle --min 0 --max 1e30 --default 1.0')
    arg2 = ArgFloat()
    arg2.fromString('--type float --name edge2 --desc The length of the second side of the triangle --min 0 --max 1e30 --default 1.0')
    arg3 = ArgFloat()
    arg3.fromString('--type float --name angle --desc The angle between the first and second sides of the triangle --min 0 --max 3.141592653589793 --default 0.7853981633974483')
    inArgList = [arg1, arg2, arg3]
    arg4 = ArgFloat()
    arg4.fromString('--type float --name edge3 --desc The length of the third side of the triangle --min 0 --max 1e30 --default 0.0')
    arg5 = ArgFloat()
    arg5.fromString('--type float --name area --desc The area of the triangle --min 0 --max 1e30 --default 0.0')
    outArgList = [arg4, arg5]
    improUiWrap(inArgList, outArgList, lawOfCosine, winTitle='GUI for law of cosine')

    # example 2: triangle area given three x-y coordinates of the vertices
    arg1 = ArgFloat()
    arg1.fromString('--type float --name x1  --desc The x-coordinate of the first vertex of the triangle --min -1e30 --max 1e30 --default 0.0')
    arg2 = ArgFloat()
    arg2.fromString('--type float --name y1  --desc The y-coordinate of the first vertex of the triangle --min -1e30 --max 1e30 --default 0.0')
    arg3 = ArgFloat()
    arg3.fromString('--type float --name x2  --desc The x-coordinate of the second vertex of the triangle --min -1e30 --max 1e30 --default 1.0')
    arg4 = ArgFloat()
    arg4.fromString('--type float --name y2  --desc The y-coordinate of the second vertex of the triangle --min -1e30 --max 1e30 --default 0.0')
    arg5 = ArgFloat()
    arg5.fromString('--type float --name x3  --desc The x-coordinate of the third vertex of the triangle --min -1e30 --max 1e30 --default 0.0')
    arg6 = ArgFloat()
    arg6.fromString('--type float --name y3  --desc The y-coordinate of the third vertex of the triangle --min -1e30 --max 1e30 --default 1.0')
    inArgList = [arg1, arg2, arg3, arg4, arg5, arg6]
    arg7 = ArgFloat()
    arg7.fromString('--type float --name area --desc The area of the triangle --min 0 --max 1e30 --default 0.0')
    outArgList = [arg7]
    improUiWrap(inArgList, outArgList, areaOfTriangle, winTitle='GUI for area of triangle')

