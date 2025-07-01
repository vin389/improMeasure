import tkinter as tk
from tkinter import filedialog

def inputdlg(prompt: list, name: str='Input Dialog', numlines: int =1, defaultanswer=[]):
    """
    ThiAs function mimics matlab function inputdlg but only supports limited 
    functionality. Only arguments prompt, name, and defaultanswer are supported.
    The arguments numlines and other arguments of matlab inputdlg are
    not supported yet. 

    Example:
        prompt=['Enter the matrix size for x^2:','Enter the colormap name:'];
        name='Input for Peaks function';
        numlines = 1;
        defaultanswer = ['20', 'hsv']
        answers = inputdlg(prompt, name, numlines, defaultanswer)
        if len(answers) > 0:
            print("The inputs are:")
            print(answers)
        else:
            print("Input dialog is cancelled.")
    """
    if type(prompt) == str:
        prompt = [prompt]    
    answers = []
    edits = []
    window = tk.Tk()
    window.title(name)
    nrow = len(prompt)
    for i in range(nrow):
        frame = tk.Frame(master=window, relief=tk.RAISED, borderwidth = 0)
        frame.grid(row=i, column=0)
        label = tk.Label(master=frame, text=prompt[i])
        label.pack()
        #
        frame = tk.Frame(master=window, relief=tk.RAISED, borderwidth = 1)
        frame.grid(row=i, column=1)
        edits.append(tk.Entry(master=frame, width=100))
        if i < len(defaultanswer): 
            edits[i].insert(0, defaultanswer[i])
        edits[i].pack()
    frame = tk.Frame(master=window, relief=tk.RAISED, borderwidth = 1)
    frame.grid(row=nrow, column=0)
    bt1 = tk.Button(master=frame, text='OK')
    bt1.pack()
    frame = tk.Frame(master=window, relief=tk.RAISED, borderwidth = 1)
    frame.grid(row=nrow, column=1)
    bt2 = tk.Button(master=frame, text='Cancel')
    bt2.pack()
    def eventOK(e):
        for i in range(nrow):
            answers.append(edits[i].get())
        window.destroy()
        window.quit()
    def eventCancel(e):
        answers = []
        window.destroy()
        window.quit()
    bt1.bind('<Button>', eventOK)
    bt2.bind('<Button>', eventCancel)
    window.mainloop()
    return answers
