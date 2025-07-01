# inputdlg2.py - Enhanced Multi-type Input Dialog using Tkinter
# -------------------------------------------------------------
# This module defines a Python function `inputdlg2` that mimics and extends
# the functionality of MATLAB's `inputdlg`, using the Tkinter GUI library.
#
# Features:
# - Supports multiple input types: int, float, string, file, dir, listbox
# - For numeric inputs (int, float), validates range and changes text color (blue if valid, red if invalid)
# - For file and directory inputs, validates existence and supports file dialogs
# - For listbox input, replaces entry with a listbox
# - Tooltips: show helpful hints when hovering over prompts
# - Safe to call from within existing Tkinter apps (uses Toplevel)
#
# Usage:
#   prompts = ['Enter age:', 'Temperature:', 'Your name:', 'Choose file:', 'Select dir:', 'Pick one:']
#   datatypes = ['int 0 99', 'float -273.15 inf', 'string', 'file', 'dir', 'listbox Apple Banana Cherry']
#   tooltips = ['Age between 0 and 99', 'Must be above absolute zero', 'Any string',
#               'Browse for a file', 'Browse for a directory', 'Choose from the list']
#   result = inputdlg2(prompts, datatypes, tooltips, title="Custom Input")
#   print(result)
#
# Supported datatypes:
#   - 'int min max'      : Integer between min and max
#   - 'float min max'    : Float between min and max
#   - 'string' / 'text'  : Any string input
#   - 'file'             : Valid path to existing file (with Browse...)
#   - 'dir'              : Valid path to existing directory (with Browse...)
#   - 'listbox item1 item2 ...' : User selects one from list of items

import tkinter as tk
from tkinter import filedialog
import os

class InputDialog2:
    active_tooltip = None  # shared reference to active tooltip

    def __init__(self, prompts, datatypes, tooltips=None, title="Input Dialog"):
        self.prompts = prompts
        self.datatypes = datatypes
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
                if int(args[0]) <= v <= int(args[1]):
                    entry.config(fg='blue')
                else:
                    entry.config(fg='red')
            elif kind == 'float':
                v = float(val)
                if float(args[0]) <= v <= float(args[1]):
                    entry.config(fg='blue')
                else:
                    entry.config(fg='red')
            elif kind == 'file':
                entry.config(fg='blue' if os.path.isfile(val) else 'red')
            elif kind == 'dir':
                entry.config(fg='blue' if os.path.isdir(val) else 'red')
        except:
            entry.config(fg='red')

    def browse(self, entry, kind):
        if kind == 'file':
            path = filedialog.askopenfilename()
        elif kind == 'dir':
            path = filedialog.askdirectory()
        else:
            return
        if path:
            entry.delete(0, tk.END)
            entry.insert(0, path)
            entry.event_generate('<KeyRelease>')

    def add_tooltip(self, widget, text):
        tooltip = tk.Toplevel(widget)
        tooltip.withdraw()
        tooltip.overrideredirect(True)
        label = tk.Label(tooltip, text=text, background="#ffffe0", relief='solid', borderwidth=1)
        label.pack()

        def show_tooltip(event):
            if InputDialog2.active_tooltip and InputDialog2.active_tooltip != tooltip:
                InputDialog2.active_tooltip.withdraw()
            x = event.x_root + 20
            y = event.y_root + 10
            tooltip.geometry(f"+{x}+{y}")
            tooltip.deiconify()
            InputDialog2.active_tooltip = tooltip

        def enter(event):
            if hasattr(self, "tooltip_after_id") and self.tooltip_after_id:
                widget.after_cancel(self.tooltip_after_id)
            self.tooltip_after_id = widget.after(1000, lambda: show_tooltip(event))

        def leave(event):
            if hasattr(self, "tooltip_after_id") and self.tooltip_after_id:
                widget.after_cancel(self.tooltip_after_id)
                self.tooltip_after_id = None
            tooltip.withdraw()
            if InputDialog2.active_tooltip == tooltip:
                InputDialog2.active_tooltip = None

        widget.bind("<Enter>", enter)
        widget.bind("<Leave>", leave)

    def show(self):
        self.result = None

        # Determine if we already have a Tk root
        if tk._default_root:
            self.root = tk.Toplevel()
            self.root.transient(tk._default_root)
        else:
            hidden_root = tk.Tk()
            hidden_root.withdraw()
            self.root = tk.Toplevel(hidden_root)

        self.root.title(self.title)
        self.root.resizable(False, False)
        self.root.grab_set()  # modal behavior

        frame = tk.Frame(self.root, padx=10, pady=10)
        frame.pack()

        for i, (prompt, dtype, tooltip) in enumerate(zip(self.prompts, self.datatypes, self.tooltips)):
            label = tk.Label(frame, text=prompt)
            label.grid(row=i, column=0, sticky='w')
            if tooltip:
                self.add_tooltip(label, tooltip)

            kind, args = self.parse_datatype(dtype)

            if kind == 'listbox':
                lb = tk.Listbox(frame, height=min(5, len(args)), exportselection=False)
                for item in args:
                    lb.insert(tk.END, item)
                lb.grid(row=i, column=1, padx=5, pady=2)
                self.entries.append(lb)
            else:
                entry = tk.Entry(frame, width=40)
                entry.grid(row=i, column=1, padx=5, pady=2)
                self.entries.append(entry)

                if kind in ['int', 'float', 'file', 'dir']:
                    entry.bind('<KeyRelease>', lambda e, ent=entry, k=kind, a=args: self.validate_input(ent, k, a))

                if kind in ['file', 'dir']:
                    browse_button = tk.Button(frame, text="Browse...", command=lambda e=entry, k=kind: self.browse(e, k))
                    browse_button.grid(row=i, column=2, padx=5)

        button_frame = tk.Frame(frame)
        button_frame.grid(row=len(self.prompts), columnspan=3, pady=10)
        tk.Button(button_frame, text="OK", width=10, command=self.on_ok).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Cancel", width=10, command=self.root.destroy).pack(side=tk.LEFT, padx=5)

        self.root.wait_window()
        return self.result

    def on_ok(self):
        results = []
        for entry, dtype in zip(self.entries, self.datatypes):
            kind, _ = self.parse_datatype(dtype)
            if kind == 'listbox':
                sel = entry.curselection()
                results.append(entry.get(sel[0]) if sel else '')
            else:
                results.append(entry.get())
        self.result = results
        self.root.destroy()

def inputdlg2(prompts, datatypes, tooltips=None, title="Input Dialog"):
    dlg = InputDialog2(prompts, datatypes, tooltips, title)
    return dlg.show()

if __name__ == '__main__':
    prompts = ['Enter age:', 'Temperature:', 'Your name:', 'Choose file:', 'Select dir:', 'Pick one:']
    datatypes = ['int 0 99', 'float -273.15 inf', 'string', 'file', 'dir', 'listbox Apple Banana Cherry']
    tooltips = ['Age between 0 and 99', 'Must be above absolute zero', 'Free text name', 'Choose an existing file', 'Choose an existing folder', 'Select from list']
    result = inputdlg2(prompts, datatypes, tooltips, "Custom Input Dialog")
    print("Result:", result)
