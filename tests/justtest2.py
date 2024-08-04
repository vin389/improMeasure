
# 這個函數跳出一個 tkinter 視窗，裡面有 4 個文字框，其中前三個分別是 三角形的三個邊長。
# 這個函數根據三個邊長，判斷這三個邊長能否構成一個三角形，如果可以，則計算它的面積。
# 如果不能構成三角形，則返回 0。
# 然後把面積顯示在第四個文字框中。
#
# 這個函數的主要目的是展示如何使用 tkinter 來創建 GUI 界面，並且如何在 GUI 界面中使用文字框、按鈕等元素。
# 這個函數中使用了 tkinter 中的 Entry、Button、Label 這三個元素。
# Entry 用來顯示文字框，Button 用來顯示按鈕，Label 用來顯示標籤。
# 這個函數中還使用了 tkinter 中的 grid 函數來設置元素的位置。
# 這個函數中還使用了 tkinter 中的 messagebox 函數來彈出對話框。
# 這個函數中還使用了 tkinter 中的 StringVar 類來設置文字框的值。
# 這個函數中還使用了 tkinter 中的 mainloop 函數來運行 GUI 界面。
#
def triangleArea():
    import tkinter as tk
    from tkinter import messagebox
    from tkinter import StringVar

    def calculateArea():
        try:
            a = float(sideA.get())
            b = float(sideB.get())
            c = float(sideC.get())
            if a + b > c and a + c > b and b + c > a:
                s = (a + b + c) / 2
                area = (s * (s - a) * (s - b) * (s - c)) ** 0.5
                areaVar.set(f'{area:.2f}')
            else:
                areaVar.set('0')
                messagebox.showerror('Error', 'The sides cannot form a triangle.')
        except ValueError:
            areaVar.set('0')
            messagebox.showerror('Error', 'Please enter valid numbers.')

    root = tk.Tk()
    root.title('Triangle Area Calculator')
    sideA = StringVar()
    sideB = StringVar()
    sideC = StringVar()
    areaVar = StringVar()
    tk.Label(root, text='Side A:').grid(row=0, column=0)
    tk.Entry(root, textvariable=sideA).grid(row=0, column=1)
    tk.Label(root, text='Side B:').grid(row=1, column=0)
    tk.Entry(root, textvariable=sideB).grid(row=1, column=1)
    tk.Label(root, text='Side C:').grid(row=2, column=0)
    tk.Entry(root, textvariable=sideC).grid(row=2, column=1)
    tk.Button(root, text='Calculate', command=calculateArea).grid(row=3, column=0)
    tk.Label(root, text='Area:').grid(row=3, column=1)
    tk.Entry(root, textvariable=areaVar, state='readonly').grid(row=3, column=2)
    root.mainloop()

# 一個主程式，用以示範如何使用這個函數。
if __name__ == '__main__':
    triangleArea()

    