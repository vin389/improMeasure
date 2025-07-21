# inputdlg3.py - Enhanced Multi-type Input Dialog using Tkinter
# -------------------------------------------------------------
# This module defines a Python function `inputdlg3` that mimics and extends
# the functionality of MATLAB's `inputdlg`, using the Tkinter GUI library.
#
# Features:
# - Supports multiple input types: int, float, string, strings, file, files, dir, listbox
# - For numeric inputs (int, float), validates range and changes text color (blue if valid, red if invalid)
# - For file and directory inputs, validates existence and supports file dialogs
# - For listbox input, replaces entry with a listbox
# - Tooltips: show helpful hints when hovering over prompts
# - Safe to call from within existing Tkinter apps (uses Toplevel)
#
# Supported datatypes:
#   - 'int min max'        : Integer between min and max. It will append an integer in returned list. None if input cannot be converted to an integer.
#   - 'float min max'      : Float between min and max. It will append a float in returned list. float('nan') if input cannot be converted to a float.
#   - 'string' / 'text'    : Any single-line string input. It will append a string in returned list.
#   - 'strings'            : Multi-line string input (returned as list of strings). I will append a list of strings in returned list.
#   - 'file'               : Valid path to existing file (with Browse...). It will append a string in returned list.
#   - 'filew'              : Valid path to new file (with Browse...). It will append a string in returned list.
#   - 'files'              : Select multiple files (returned as list of paths). It will append a list of strings in returned list.
#   - 'dir'                : Valid path to existing directory (with Browse...). It will append a string in returned list.
#   - 'listbox item1 ...'  : User selects one from list of items. It will append a list of selected indices in returned list. E.g., [0, 2] means the first and third items are selected.
#   - 'array dim0 dim1'    : Input a 1D array of numbers, reshaped to 2D array of shape (dim0, dim1). It will append a 2D float numpy array of shape (dim0, dim1) in returned list. If fails, returns np.array([]) (empty array).
#   - 'image'              : Select an image file or a video file followed by a frame index
# Example usages:
#   1. Inputs for name and age:
#        prompts = ['Your name:', 'Your age:']
#        datatypes = ['string', 'int 0 120']
#        title = 'Input your data'
#        result = inputdlg3(prompts, datatypes, title=title)
#
#   2. Multi-line biography:
#        prompts = ['Tell us about yourself:']
#        datatypes = ['strings']
#        result = inputdlg3(prompts, datatypes)
#
#   3. File selection:
#        prompts = ['Select a file:']
#        datatypes = ['file']
#        result = inputdlg3(prompts, datatypes)
#
#   4. Multiple files selection:
#        prompts = ['Choose files to process:']
#        datatypes = ['files']
#        result = inputdlg3(prompts, datatypes)
#
#   5. Directory selection:
#        prompts = ['Select output directory:']
#        datatypes = ['dir']
#        result = inputdlg3(prompts, datatypes)
#
#   6. Listbox example:
#        prompts = ['Pick your favorite color:']
#        datatypes = ['listbox Red Green Blue']
#        result = inputdlg3(prompts, datatypes)
#
#   7. Temperature with tooltip:
#        prompts = ['Enter temperature (C):']
#        datatypes = ['float -50 100']
#        initvalues = ['25.0']  # default value
#        tooltips = ['Between -50 and 100']
#        result = inputdlg3(prompts, datatypes, initvalues, tooltips)
#
#   8. Input name and select file:
#        prompts = ['Enter your name:', 'Select a config file:']
#        datatypes = ['string', 'file']
#        result = inputdlg3(prompts, datatypes)
#
#   9. Logging form with attachments:
#        prompts = ['Log message:', 'Attach files:']
#        datatypes = ['strings', 'files']
#        result = inputdlg3(prompts, datatypes)
#
#  10. Choose day and provide note:
#        prompts = ['Pick a day:', 'Note:']
#        datatypes = ['listbox Mon Tue Wed Thu Fri', 'strings']
#        result = inputdlg3(prompts, datatypes)
#
#  11. Directory with tooltip:
#        prompts = ['Where to save files?']
#        datatypes = ['dir']
#        tooltips = ['Pick an existing folder']
#        result = inputdlg3(prompts, datatypes, tooltips=tooltips)
#
#  12. Basic input with title:
#        prompts = ['Username:', 'Password:']
#        datatypes = ['string', 'string']
#        title = 'Login Information'
#        result = inputdlg3(prompts, datatypes, title=title)
#
#  13. Basic input with title:
#        prompts = ['Username:', 'Matrix 2 3:']
#        datatypes = ['string', 'array 2 3']
#        title = 'Login Information'
#        result = inputdlg3(prompts, datatypes, title=title)

import tkinter as tk
from tkinter import filedialog
import os
import numpy as np

class InputDialog3:
    active_tooltip = None

    def __init__(self, prompts, datatypes, initvalues, tooltips=None, title="Input Dialog"):
        self.prompts = prompts
        self.datatypes = datatypes
        self.initvalues = initvalues if initvalues else ["" for _ in prompts]
        # make initvalues a list that has the same length of prompts, and fill Nones if not provided
        if initvalues is None:
            initvalues = [None] * len(prompts)
        if len(initvalues) < len(prompts):
            self.initvalues += [None] * (len(prompts) - len(initvalues))
        elif len(initvalues) > len(prompts):
            self.initvalues = initvalues[:len(prompts)]
        self.tooltips = tooltips if tooltips else ["" for _ in prompts]
        self.title = title
        self.entries = []
        self.values = []

    def parse_datatype(self, dtype):
        parts = dtype.strip().split()
        kind = parts[0].lower()
        args = parts[1:]
        return kind, args

    def validate_input(self, entry, kind, args):
        val = entry.get()
        try:
            if kind == 'int':
                v = int(val)
                entry.config(fg='blue' if int(args[0]) <= v <= int(args[1]) else 'red')
            elif kind == 'float':
                v = float(val)
                entry.config(fg='blue' if float(args[0]) <= v <= float(args[1]) else 'red')
            elif kind == 'file':
                entry.config(fg='blue' if os.path.isfile(val) else 'red')
            elif kind == 'dir':
                entry.config(fg='blue' if os.path.isdir(val) else 'red')
            elif kind == 'array':
#                nums = list(map(float, val.strip().split()))
#                import er
#                str_list = re.split(r'[\s,]+', val.strip())
                # replace [ ] , \t \n \r with whitespace
                val = val.strip()
                val = val.replace('[', ' ')
                val = val.replace(']', ' ')
                val = val.replace(',', ' ')
                val = val.replace(';', ' ')
                str_list = val.split()
                dim0, dim1 = int(args[0]), int(args[1])
#                np.array(nums).reshape((dim0, dim1))
                np.array(str_list, dtype=float).reshape((dim0, dim1))
                entry.config(fg='blue')
            elif kind == 'image':
                str_list = val.strip().split()
                # if the input has single string, it is an image file, try to open it
                if len(str_list) == 1:
                    if os.path.isfile(str_list[0]):
                        import cv2
                        img = cv2.imread(str_list[0])
                        if img is not None:
                            entry.config(fg='blue')
                        else:
                            entry.config(fg='red')
                    else:
                        entry.config(fg='red')
                # if the input has two strings, it is a video file followed by a frame index (1-based)
                if len(str_list) == 2:
                    entry.config(fg='red')
                    if os.path.isfile(str_list[0]):
                        frame_index = int(str_list[1])
                        if frame_index >= 0:
                            import cv2
                            cap = cv2.VideoCapture(str_list[0])
                            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                            if 0 < frame_index <= total_frames:
                                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index - 1)
                                ret, img = cap.read()
                                if ret and img is not None:
                                    cap.release()
                                    entry.config(fg='blue')
        except:
            entry.config(fg='red')

    def browse(self, entry, kind):
        if kind == 'file':
            path = filedialog.askopenfilename()
            # if dialog is cancelled and path is selected 
            if path:
                entry.delete(0, tk.END)
                entry.insert(0, path)
        if kind == 'filew':
            # the file can be new.
            path = filedialog.asksaveasfilename()
            # if dialog is cancelled and path is selected 
            if path:
                entry.delete(0, tk.END)
                entry.insert(0, path)
        elif kind == 'dir':
            path = filedialog.askdirectory()
            if path:
                entry.delete(0, tk.END)
                entry.insert(0, path)
        elif kind == 'files':
            paths = filedialog.askopenfilenames()
            entry.delete("1.0", tk.END)
            entry.insert(tk.END, "\n".join(paths))
        elif kind == 'image':
            # when user clicks "Browse..." for image, show a file dialog to select an image file 
            # or a video file followed by asking for a frame index through simple tk dialog
            path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"),
                                                         ("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")])
            # if user cancels
            if not path:
                return
            # if the selected file is a video, ask for a frame index
            if path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                # if user selects a video file, ask for a frame index
                frame_index = tk.simpledialog.askinteger("Frame Index", "Enter frame index (1-based):", minvalue=1)
                # append the frame_index as a string after the video path
                entry.delete(0, tk.END)
                entry.insert(0, path + (" %d" % frame_index if frame_index is not None else -1))
            else:
                entry.delete(0, tk.END)
                entry.insert(0, path)

        else:
            return
        # validate the input after browsing by generating a KeyRelease event
        entry.event_generate('<KeyRelease>')
#        self.validate_input(self, entry, kind, [])

    def add_tooltip(self, widget, text):
        tooltip = tk.Toplevel(widget)
        tooltip.withdraw()
        tooltip.overrideredirect(True)
        label = tk.Label(tooltip, text=text, background="#ffffe0", relief='solid', borderwidth=1)
        label.pack()

        def show_tooltip(event):
            if InputDialog3.active_tooltip and InputDialog3.active_tooltip != tooltip:
                InputDialog3.active_tooltip.withdraw()
            x = event.x_root + 20
            y = event.y_root + 10
            tooltip.geometry(f"+{x}+{y}")
            tooltip.deiconify()
            InputDialog3.active_tooltip = tooltip

        def enter(event):
            if hasattr(self, "tooltip_after_id") and self.tooltip_after_id:
                widget.after_cancel(self.tooltip_after_id)
            self.tooltip_after_id = widget.after(1000, lambda: show_tooltip(event))

        def leave(event):
            if hasattr(self, "tooltip_after_id") and self.tooltip_after_id:
                widget.after_cancel(self.tooltip_after_id)
                self.tooltip_after_id = None
            tooltip.withdraw()
            if InputDialog3.active_tooltip == tooltip:
                InputDialog3.active_tooltip = None

        widget.bind("<Enter>", enter)
        widget.bind("<Leave>", leave)

    def show(self):
        self.result = None
        self.hidden_root = None

        if tk._default_root:
            self.root = tk.Toplevel()
            self.root.transient(tk._default_root)
        else:
            self.hidden_root = tk.Tk()
            self.hidden_root.withdraw()
            self.root = tk.Toplevel(self.hidden_root)
            # make self.root on top of any other windows
#            self.root.attributes('-topmost', True)          

        self.root.title(self.title)
        self.root.resizable(False, False)
        self.root.grab_set()

        frame = tk.Frame(self.root, padx=10, pady=10)
        frame.pack()

        for i, (prompt, dtype, initvalue, tooltip) in enumerate(zip(self.prompts, self.datatypes, self.initvalues, self.tooltips)):
            label = tk.Label(frame, text=prompt)
            label.grid(row=i, column=0, sticky='w', padx=5, pady=2)
            # make the label width fit the longest prompt, and make text align left
            label.config(anchor='w')
            label.config(width=max(10, len(prompt) + 2))
            if tooltip:
                self.add_tooltip(label, tooltip)

            kind, args = self.parse_datatype(dtype)

            if kind == 'listbox':
                lb = tk.Listbox(frame, height=min(5, len(args)), exportselection=False, selectmode=tk.EXTENDED)
                for item in args:
                    lb.insert(tk.END, item)
                lb.grid(row=i, column=1, sticky='w', padx=5, pady=2)
                self.entries.append(lb)
                # set initial selection if provided (e.g., initvalue = [0, 2] means select first and third items)
                if initvalue and isinstance(initvalue, list):
                    for idx in initvalue:
                        if 0 <= idx < lb.size():
                            lb.select_set(idx)
            elif kind in ['strings', 'files']:
                text = tk.Text(frame, height=3, width=40)
                text.grid(row=i, column=1, sticky='w', padx=5, pady=2)
                self.entries.append(text)
                # set initvalue for text input, e.g., initvalue = "line1\nline2" or initvalue = ["line1", "line2"]
                if initvalue:
                    text.insert(tk.END, "\n".join(initvalue) if isinstance(initvalue, list) else initvalue)
                if kind == 'files':
                    browse_button = tk.Button(frame, text="Browse...", command=lambda e=text, k=kind: self.browse(e, k))
                    browse_button.grid(row=i, column=2, sticky='w', padx=5)
            else:
                entry = tk.Entry(frame, width=40)
                entry.grid(row=i, column=1, sticky='w', padx=5, pady=2)
                self.entries.append(entry)
                # set initial value if provided
                if initvalue:
                    entry.insert(0, initvalue)
                # validate input on key release
                if kind in ['int', 'float', 'file', 'dir', 'array', 'image']:
                    entry.bind('<KeyRelease>', lambda e, ent=entry, k=kind, a=args: self.validate_input(ent, k, a))
                if kind in ['file', 'filew', 'dir', 'image']:
                    browse_button = tk.Button(frame, text="Browse...", command=lambda e=entry, k=kind: self.browse(e, k))
                    browse_button.grid(row=i, column=2, padx=5)
                if kind in ['image']:
                    # add a button to show the image or pick POIs
                    # the command function will use cv2.imread to read the image and display it by imshow3
                    # or if it is a video file, it will read the frame specified by the index
                    # and display it by imshow3
                    def imshow_command(entry, kind):
                        import cv2
                        val = entry.get().strip()
                        if val:
                            str_list = val.split()
                            if len(str_list) == 1:
                                # single image file
                                img = cv2.imread(str_list[0])
                                if img is not None:
                                    from imshow3 import imshow3
                                    imshow3(img=img, winname="Image from Dialog: %s" % os.path.basename(str_list[0]), winmax=(640,640))
                                else:
                                    print(f"Could not read image from {str_list[0]}.")
                            elif len(str_list) == 2:
                                # video file with frame index
                                cap = cv2.VideoCapture(str_list[0])
                                frame_index = int(str_list[1]) - 1
                                if cap.isOpened():
                                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                                    if 0 <= frame_index < total_frames:
                                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                                        ret, img = cap.read()
                                        if ret and img is not None:
                                            from imshow3 import imshow3
                                            imshow3(img=img, winname="Video Frame from Dialog: %s Frame %d" % (str_list[0], frame_index+1), winmax=(1024,768))
                                    else:
                                        print(f"Frame index {frame_index + 1} out of range for video with {total_frames} frames.")
                    image_show_button = tk.Button(frame, text="Imshow/PickPOIs", command=lambda e=entry, k=kind: imshow_command(e,k))
                    image_show_button.grid(row=i, column=3, padx=5)


        button_frame = tk.Frame(frame)
        button_frame.grid(row=len(self.prompts), column=0, columnspan=3, sticky='w', pady=10)
        tk.Button(button_frame, text="OK", width=10, command=self.on_ok).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Cancel", width=10, command=self.root.destroy).pack(side=tk.LEFT, padx=5)

        self.root.wait_window()

        # Clean up the hidden root if we created one
        if self.hidden_root:
            self.hidden_root.destroy()

        return self.result

    def on_ok(self):
        results = []
        for entry, dtype in zip(self.entries, self.datatypes):
            kind, args = self.parse_datatype(dtype)
            if kind == 'listbox':
                # set sels as a list of the selected index(es) in the listbox
                # E.g., sels = [0, 2] means the first and third items are selected
                # E.g., sels = [] means no item is selected
                try:
                    # type(sel) is 
                    sel = entry.curselection()
                    results.append(list(sel))
                except:
                    results.append([])
            elif kind == 'int':
                try:
                    results.append(int(entry.get()))
                except:
                    results.append(None)
            elif kind == 'float':
                try:
                    results.append(float(entry.get()))
                except:
                    results.append(float('nan'))
            elif kind in ['strings', 'files']:
                raw = entry.get("1.0", tk.END).strip()
                results.append(raw.splitlines())
            elif kind == 'array':
                # try to convert input to a 2D numpy array
                try:
                    import re
                    str_list = re.split(r'[\s,]+', entry.get().strip())
                    dim0, dim1 = int(args[0]), int(args[1])
                    arr = np.array(str_list, dtype=float).reshape((dim0, dim1))
                    results.append(arr)
                except ValueError:
                    results.append(np.array([]))  # return empty array on error
            else:
                results.append(entry.get())
        self.result = results
        self.root.destroy()

def inputdlg3(prompts, datatypes, initvalues=None, tooltips=None, title="Input Dialog"):
    dlg = InputDialog3(prompts, datatypes, initvalues, tooltips, title)
    return dlg.show()

if __name__ == '__main__':
    prompts = ['Name:', 
               'Age (0-99):', 
               'Height (cm, a float, 40-250)', 
               'Brief intro. (multiple lines)', 
               'A file:', 
               'Multiple files:', 
               'A directory:', 
               'A zodiac', 
               'Matrix (2x3):',
               'An image or a video frame:']
    datatypes = ['string', 
                 'int 0 99', 
                 'float 40 250.', 
                 'strings', 
                 'file', 
                 'files', 
                 'dir', 
                 'listbox Caprecorn Aquarium Pisces Aries Taurus Gemini Cancer Leo Virgo Libra Scorpio Sagittarius', 
                 'array 2 3',
                 'image']
    initvalues = ['John Doe', 
                '25', 
                '175.5', 
                # introduce John Doe with a multi-line string
                'John Doe is a software engineer.\nHe loves coding and enjoys solving problems.\nIn his free time, he likes to read books and play video games.',
                './iputdlg3.py',  # file input, initially empty
                '',  # files input, initially empty
                './',  # dir input, initially empty
                [0,2,4],  # default selection for listbox
                '1.0 2.0 3.0,4.0 5.0 6.0', # example matrix input
                'c:/temp/video.mp4 1']  # example image input, a video file with frame index 1
    tooltips = ['Name, allows dot and -', 
                'Age between 0 and 99', 
                'Height between 40.0 and 250.0', 
                'Sentences, multi-line strings', 
                'A file (blue:exists, red:not exist)', 
                'Multiple files',
                'a directory (blue:exists, red:not exist', 
                'Select one. It returns the string', 
                '6 numbers separated by spaces or commas to represent a 2 3 matrix',
                'Select an image or a video file followed by entering a frame index (1-based)']
    result = inputdlg3(prompts=prompts, datatypes=datatypes, initvalues=initvalues, tooltips=tooltips, title="Demo with inputdlg3")
    print("Result:", result)
