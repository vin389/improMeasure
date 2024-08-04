import tkinter as tk
from tkinter import ttk

def celsius_to_fahrenheit(celsius):
    return (celsius * 9/5) + 32

def on_celsius_change(*args):
    try:
        celsius_value = float(celsius_var.get())
        fahrenheit_value = celsius_to_fahrenheit(celsius_value)
        fahrenheit_var.set(f"{fahrenheit_value:.2f}")
    except ValueError:
        fahrenheit_var.set("")

def show_tooltip(event):
    x = event.widget.winfo_rootx() + 20
    y = event.widget.winfo_rooty() + 20
    tooltip.place(x=x, y=y)

def hide_tooltip(event):
    tooltip.place_forget()

# Create the main window
root = tk.Tk()
root.title("Celsius to Fahrenheit Converter")

# Create StringVar variables to hold the text entries
celsius_var = tk.StringVar()
fahrenheit_var = tk.StringVar()

# Set up the trace on the Celsius variable to update the Fahrenheit value
celsius_var.trace_add("write", on_celsius_change)

# Create and grid the widgets
celsius_label = ttk.Label(root, text="Celsius:")
celsius_label.grid(column=0, row=0, padx=5, pady=5)

celsius_entry = ttk.Entry(root, textvariable=celsius_var)
celsius_entry.grid(column=1, row=0, padx=5, pady=5)

fahrenheit_label = ttk.Label(root, text="Fahrenheit:")
fahrenheit_label.grid(column=0, row=1, padx=5, pady=5)

fahrenheit_entry = ttk.Entry(root, textvariable=fahrenheit_var, state='readonly')
fahrenheit_entry.grid(column=1, row=1, padx=5, pady=5)

# Create a tooltip label
tooltip = tk.Label(root, text="Enter temperature in Celsius", bg="yellow", bd=1, relief="solid")

# Bind events to show and hide the tooltip
celsius_entry.bind("<Enter>", show_tooltip)
celsius_entry.bind("<Leave>", hide_tooltip)

# Run the Tkinter event loop
root.mainloop()
