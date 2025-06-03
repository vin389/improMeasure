import tkinter as tk
from tkinter import StringVar
import time

# ImproToolTip class is used to create tooltips for widgets in a Tkinter application.
# It shows a tooltip with a specified text when the mouse hovers over the widget for a certain delay.
# It is easy to use and can be attached to Tkinter widget.
# The usage is simple: Giving widget object, tooltip text, and delay in milliseconds.
# The return value tooltip is not used normally, so it can be ignored.
# Example usage: 
#   tooltip = ImproToolTip(widget, "This is a tooltip text", delay=200)
#
# This file also gives a Fahrenheit-Celsius converter to demonstrate how to use it.
# 

class ImproToolTip:
    def __init__(self, widget, text, delay=200):
        self.widget = widget
        self.text = text
        self.delay = delay
        self.tipwindow = None
        self.id = None
        self.enter_time = None
        self.widget.bind("<Enter>", self.schedule)
        self.widget.bind("<Leave>", self.hide)

    def schedule(self, event=None):
        self.enter_time = time.time()
        self.id = self.widget.after(self.delay, self.show)

    def show(self):
        if self.tipwindow or (time.time() - self.enter_time) * 1000 < self.delay:
            return
        x, y, cx, cy = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 25
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        label = tk.Label(tw, text=self.text, background="yellow", relief='solid', borderwidth=1)
        label.pack()

        tw.wm_geometry(f"+{x}+{y}")

    def hide(self, event=None):
        if self.id:
            self.widget.after_cancel(self.id)
            self.id = None
        if self.tipwindow:
            self.tipwindow.destroy()
            self.tipwindow = None

class TempConverterApp:
    def __init__(self, root):
        self.root = root
        root.title("Celsius-Fahrenheit Converter")

        self.saved_celsius = ""
        self.saved_fahrenheit = ""

        self.var_cel = StringVar()
        self.var_fah = StringVar()

        self.var_cel.trace_add("write", self.on_celsius_change)
        self.var_fah.trace_add("write", self.on_fahrenheit_change)

        self.is_updating = False

        # Celsius Entry
        tk.Label(root, text="Celsius:").grid(row=0, column=0, padx=5, pady=5, sticky='e')
        self.ed_cel = tk.Entry(root, textvariable=self.var_cel, name="ed_cel")
        self.ed_cel.grid(row=0, column=1)
        ImproToolTip(self.ed_cel, "Enter temperature in Celsius.\nF = C * 9 / 5 + 32")

        # Fahrenheit Entry
        tk.Label(root, text="Fahrenheit:").grid(row=1, column=0, padx=5, pady=5, sticky='e')
        self.ed_fah = tk.Entry(root, textvariable=self.var_fah, name="ed_fah")
        self.ed_fah.grid(row=1, column=1)
        ImproToolTip(self.ed_fah, "Enter temperature in Fahrenheit.\nC = (F - 32) * 5 / 9")

        # Buttons
        self.bt_clear = tk.Button(root, text="Clear", name="bt_clear", command=self.clear)
        self.bt_clear.grid(row=2, column=0, padx=5, pady=5)
        ImproToolTip(self.bt_clear, "Clear both fields.")

        self.bt_save = tk.Button(root, text="Save", name="bt_save", command=self.save)
        self.bt_save.grid(row=2, column=1, padx=5, pady=5)
        ImproToolTip(self.bt_save, "Save current values for later use.")

        self.bt_load = tk.Button(root, text="Load", name="bt_load", command=self.load)
        self.bt_load.grid(row=2, column=2, padx=5, pady=5)
        ImproToolTip(self.bt_load, "Load saved values.")

    def on_celsius_change(self, *args):
        if self.is_updating:
            return
        try:
            c = float(self.var_cel.get())
            f = c * 9 / 5 + 32
            self.is_updating = True
            self.var_fah.set(f"{f:.2f}")
        except ValueError:
            if self.var_cel.get() == "":
                self.is_updating = True
                self.var_fah.set("")
        finally:
            self.is_updating = False

    def on_fahrenheit_change(self, *args):
        if self.is_updating:
            return
        try:
            f = float(self.var_fah.get())
            c = (f - 32) * 5 / 9
            self.is_updating = True
            self.var_cel.set(f"{c:.2f}")
        except ValueError:
            if self.var_fah.get() == "":
                self.is_updating = True
                self.var_cel.set("")
        finally:
            self.is_updating = False

    def clear(self):
        self.var_cel.set("")
        self.var_fah.set("")

    def save(self):
        self.saved_celsius = self.var_cel.get()
        self.saved_fahrenheit = self.var_fah.get()

    def load(self):
        self.var_cel.set(self.saved_celsius)
        self.var_fah.set(self.saved_fahrenheit)

if __name__ == "__main__":
    root = tk.Tk()
    app = TempConverterApp(root)
    root.mainloop()