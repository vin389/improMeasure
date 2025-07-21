import tkinter as tk
from video_to_pictures import VideoToPicturesGUI
from pictures_to_video import PicturesToVideoGUI
from imshow3 import imshow3, run_imshow3_gui
from template_match_subpixels import run_gui_template_match_subpixels_image_sequence

# Tooltip class
class Tooltip:
    def __init__(self, widget, text, delay=1000):
        self.widget = widget
        self.text = text
        self.delay = delay
        self.tipwindow = None
        self.id = None
        self.widget.bind("<Enter>", self.schedule)
        self.widget.bind("<Leave>", self.hide_tip)

    def schedule(self, event=None):
        self.unschedule()
        self.id = self.widget.after(self.delay, self.show_tip)

    def unschedule(self):
        if self.id:
            self.widget.after_cancel(self.id)
            self.id = None

    def show_tip(self):
        if self.tipwindow or not self.text:
            return
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 40
        y += self.widget.winfo_rooty() + 20
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(tw, text=self.text, justify='left',
                         background="#ffffe0", relief='solid', borderwidth=1,
                         font=("tahoma", "8", "normal"))
        label.pack(ipadx=4)

    def hide_tip(self, event=None):
        self.unschedule()
        if self.tipwindow:
            self.tipwindow.destroy()
            self.tipwindow = None


class ImToolsApp:
    def __init__(self, master):
        self.master = master
        master.title("ImTools - Image Measurement Tools")

        self.button_frame = tk.Frame(master)
        self.button_frame.pack(anchor='nw', padx=10, pady=10)

        # Video to Pictures tool for extracting frames from a video
        self.add_tool_button(
            label="Video to Pictures", 
            command=self.launch_video_to_pictures, 
            tooltip_text="Convert frames in a video to image files."
        )
        # Pictures to Video tool for combining images into a video
        self.add_tool_button(
            label="Pictures to Video", 
            command=self.launch_pictures_to_video, 
            tooltip_text="Combine image files into a video file."
        )
        # Imshow3 tool for image display with zoom/pan and template picking
        self.add_tool_button(
            label="Imshow/Pick", 
            command=self.launch_imshow3, 
            tooltip_text="Show image w/ zoom/pan and picking templates."
        )
        # Image sequence template match
        self.add_tool_button(
            label="Template match (Img seq)", 
            command=self.run_template_match_imgseq,
            tooltip_text="Show image w/ zoom/pan and picking templates."
        )
        # add a new button here for any new tool
        self.add_tool_button(
            label="Exit",
            command=master.quit, 
            tooltip_text="Exit the application."
        )

    def add_tool_button(self, label, command, tooltip_text):
        button = tk.Button(self.button_frame, text=label, width=40, command=command)
        button.pack(anchor='w', pady=5)
        Tooltip(button, tooltip_text)

    def launch_video_to_pictures(self):
        tool_window = tk.Toplevel(self.master)
        VideoToPicturesGUI(tool_window)

    def launch_pictures_to_video(self):
        tool_window = tk.Toplevel(self.master)
        PicturesToVideoGUI(tool_window)

    def launch_imshow3(self):
#        tool_window = tk.Toplevel(self.master)
        run_imshow3_gui()
    def run_template_match_imgseq(self):
        from template_match_subpixels import run_gui_template_match_subpixels_image_sequence
        run_gui_template_match_subpixels_image_sequence()



if __name__ == "__main__":
    root = tk.Tk()
    app = ImToolsApp(root)
    root.mainloop()
