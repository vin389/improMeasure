"""
ImgSeq: Unified Interface for Accessing Image Sequences or Video Frames
=======================================================================

Purpose:
--------
The `ImgSeq` class provides a unified interface for accessing images from either
- A video file (e.g., MP4, AVI, MOV)
- A sequence of images (e.g., IMG03333.JPG, IMG03334.JPG, ...)

This eliminates the need to write different logic for handling video frames
(using `cv2.VideoCapture`) or loading images (using `cv2.imread`). It is designed
for use in image-based experiments and vision-based research where frames must
be accessed in a consistent, indexed manner.

Functionality Overview:
------------------------
- Consistently load and index image sources
- Support random frame access with or without keyframe dependencies
- Provide GUI support via Tkinter for file selection and interactive access

Important Methods:
------------------
- `__init__(path=None, start_index=None, count=None)`: Initialize from a file or pattern
- `get_image(index, sequential_read=False)`: Return frame/image at 0-based index
- `__len__()`: Return total number of images
- `uigetfile(...)`: Open file dialog and load a selected video/image set
- `uigetfile2(...)`: Launch GUI to browse, view info, and display images

Important Variables:
--------------------
- `self.source_type`: Either 'video' or 'images'
- `self.paths`: List of image paths (used if source is image sequence)
- `self.video_cap`: OpenCV VideoCapture object (if source is video)
- `self.total_frames`: Number of frames in source

Usage Examples:
---------------
1. Load a video:
    imgseq = ImgSeq("c:/temp/video.mp4")
    img = imgseq.get_image(0)

2. Load numbered images using a C-style format:
    imgseq = ImgSeq("c:/img%04d.jpg", 1, 10)  # img0001.jpg to img0010.jpg

3. Load images using a wildcard:
    imgseq = ImgSeq("c:/images/*.jpg")

4. Load image list from a text file:
    imgseq = ImgSeq("c:/list.txt")  # each line is an image path

5. Use file dialog to load image source:
    imgseq = ImgSeq()
    imgseq.uigetfile()

6. Use GUI to browse and view frames:
    imgseq = ImgSeq()
    imgseq.uigetfile2()

7. Get number of frames/images:
    print(len(imgseq))

8. Show 100th frame:
    img = imgseq.get_image(99)
    cv2.imshow("Frame", img)
    cv2.waitKey(0)

9. Show frame 100 with I-frame safety:
    img = imgseq.get_image(99, sequential_read=True)

10. Display only valid frames:
    if 0 <= i < len(imgseq):
        img = imgseq.get_image(i)

11. Integrate with custom GUI imshow:
    imshow3("Frame 51", imgseq.get_image(50))

12. Run the GUI twice in one session:
    imgseq = ImgSeq()
    # For first time, imgseq is empty, and uigetfile2() allows user to select a video/image sequence
    imgseq.uigetfile2()
    # For second time, imgseq retains the selected sequence info, and also allows user to update it.
    imgseq.uigetfile2()

"""

import os
import glob
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
from imshow3 import imshow3

class ImgSeq:
    def __init__(self, path=None, start_index=None, count=None):
        self.source_type = None
        self.paths = []
        self.video_cap = None
        self.video_path = None
        self.total_frames = 0

        if path and start_index is not None and count is not None:
            self._init_from_c_style(path, start_index, count)
        elif isinstance(path, str):
            self._init_from_auto(path)
        elif path is None:
            pass
        else:
            raise ValueError("Invalid constructor arguments for ImgSeq")

    def _init_from_auto(self, path):
        if path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            self._init_from_video(path)
        elif '*' in path or '?' in path:
            self._init_from_glob(path)
        elif os.path.isfile(path):
            self._init_from_filelist(path)
        else:
            raise ValueError(f"Cannot interpret input path: {path}")

    def _init_from_video(self, video_path):
        self.source_type = 'video'
        self.video_path = video_path
        self.video_cap = cv2.VideoCapture(video_path)
        if not self.video_cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")
        self.total_frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def _init_from_glob(self, pattern):
        self.source_type = 'images'
        self.paths = sorted(glob.glob(pattern))
        if not self.paths:
            raise FileNotFoundError(f"No files match pattern: {pattern}")

    def _init_from_filelist(self, list_path):
        self.source_type = 'images'
        with open(list_path, 'r') as f:
            lines = f.readlines()
        self.paths = [line.split('#')[0].strip() for line in lines if line.strip() and not line.strip().startswith('#')]
        if not self.paths:
            raise ValueError(f"No valid image paths found in {list_path}")

    def _init_from_c_style(self, pattern, start, count):
        self.source_type = 'images'
        self.paths = [pattern % (start + i) for i in range(count)]
        for p in self.paths:
            if not os.path.isfile(p):
                raise FileNotFoundError(f"Missing file: {p}")

    def __len__(self):
        return self.total_frames if self.source_type == 'video' else len(self.paths)

    def get_image(self, index, sequential_read=False):
        if self.source_type == 'video':
            if index < 0 or index >= self.total_frames:
                raise IndexError("Video frame index out of range.")
            if sequential_read:
                self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                for i in range(index + 1):
                    success, frame = self.video_cap.read()
                    if not success:
                        raise IOError(f"Failed to read frame {i}")
                return frame
            else:
                self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                success, frame = self.video_cap.read()
                if not success:
                    raise IOError(f"Failed to read frame {index}")
                return frame
        elif self.source_type == 'images':
            if index < 0 or index >= len(self.paths):
                raise IndexError("Image index out of range.")
            img = cv2.imread(self.paths[index])
            if img is None:
                raise IOError(f"Failed to load image: {self.paths[index]}")
            return img
        else:
            raise RuntimeError("Source not initialized.")

    def uigetfile(self, initial_directory=None, title="Select a video file"):
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        filetypes = [("Media/Image files", "*.mp4 *.avi *.mov *.mkv *.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        paths = filedialog.askopenfilenames(title=title, initialdir=initial_directory, filetypes=filetypes)
        if not paths:
            return ""
        video_exts = ('.mp4', '.avi', '.mov', '.mkv')
        image_exts = ('.jpg', '.jpeg', '.png', '.bmp')
        video_files = [p for p in paths if p.lower().endswith(video_exts)]
        image_files = [p for p in paths if p.lower().endswith(image_exts)]
        if video_files:
            if len(video_files) > 1:
                messagebox.showwarning("Multiple Videos", "Only the first video file will be used.")
            self._init_from_video(video_files[0])
            return video_files[0]
        elif image_files:
            self.paths = sorted(image_files)
            self.source_type = 'images'
            return image_files[0]
        else:
            messagebox.showwarning("Unsupported Format", "Selected files are not supported formats.")
            return ""

    def uigetfile2(self, initial_directory=None, title="Select a video file"):
        def browse():
            selected = self.uigetfile(initial_directory, "Select a video file or image sequence")
            if selected:
                display_paths()
                update_info()
                validate_frame(None)

        def validate_frame(event):
            try:
                idx = int(ent_frame.get()) - 1
                if 0 <= idx < len(self):
                    ent_frame.config(fg='blue')
                else:
                    ent_frame.config(fg='red')
            except:
                ent_frame.config(fg='red')

        def update_info():
            txt_info.config(state='normal')
            txt_info.delete('1.0', tk.END)
            if self.source_type == 'video' and self.video_cap:
                w = int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = self.video_cap.get(cv2.CAP_PROP_FPS)
                txt_info.insert(tk.END, f"Number of images: {len(self)}\n")
                txt_info.insert(tk.END, f"Resolution (width height): {w} {h}\n")
                txt_info.insert(tk.END, f"Frames per second: {fps:.2f}\n")
            elif self.source_type == 'images' and self.paths:
                img = cv2.imread(self.paths[0])
                if img is not None:
                    h, w = img.shape[:2]
                    txt_info.insert(tk.END, f"Number of images: {len(self)}\n")
                    txt_info.insert(tk.END, f"Resolution (width height): {w} {h}\n")
            txt_info.config(state='disabled')

        def show_image():
            try:
                idx = int(ent_frame.get()) - 1
                if 0 <= idx < len(self):
                    img = self.get_image(idx)
                    imshow3(f"Frame {idx+1}", img)
            except Exception as e:
                messagebox.showerror("Error", str(e))

        def display_paths():
            txt_paths.delete('1.0', tk.END)
            if self.source_type == 'images':
                for p in self.paths:
                    txt_paths.insert(tk.END, p + "\n")
            elif self.source_type == 'video':
                txt_paths.insert(tk.END, self.video_path + "\n")

        # Create a hidden root window to avoid issues with Toplevel
        if tk._default_root:
            # if a root exists, create a Toplevel window
            root = tk.Toplevel()
            # root.transient(tk._default_root)
            # root.attributes("-topmost", True)
        else:
            # if no root exists, create a hidden one
            # to avoid issues with Toplevel
            hidden_root = tk.Tk()
            hidden_root.withdraw()
            root = tk.Toplevel(hidden_root)

        root.title(title)
        root.geometry("700x400")

        frm_top = tk.Frame(root)
        frm_top.pack(padx=10, pady=5, fill='x')
        btn_browse = tk.Button(frm_top, text="Browse...", command=browse)
        btn_browse.pack(side='left')
        txt_paths = ScrolledText(frm_top, height=5, width=80)
        txt_paths.pack(side='left', fill='x', expand=True, padx=5)

        frm_info = tk.Frame(root)
        frm_info.pack(padx=10, pady=5, fill='x')
        txt_info = tk.Text(frm_info, height=3, width=80, fg='black', state='disabled')
        txt_info.pack(side='left', fill='x', expand=True)

        frm_mid = tk.Frame(root)
        frm_mid.pack(padx=10, pady=5, fill='x')
        tk.Label(frm_mid, text="Frame index (1-based):").pack(side='left')
        ent_frame = tk.Entry(frm_mid, width=10)
        ent_frame.pack(side='left', padx=5)
        ent_frame.bind("<KeyRelease>", validate_frame)

        frm_bot = tk.Frame(root)
        frm_bot.pack(padx=10, pady=10)
        btn_show = tk.Button(frm_bot, text="Show Image", command=show_image)
        btn_show.pack(side='left', padx=5)
        btn_close = tk.Button(frm_bot, text="Close", command=root.destroy)
        btn_close.pack(side='left', padx=5)

        display_paths()
        update_info()
#        root.mainloop()
        root.wait_window()
        print("In uigetfile2(): Image sequence initialized.")
        


if __name__ == '__main__':
    root = tk.Tk()
    root.withdraw()  # hide the root window

    seq = ImgSeq()

    seq.uigetfile2(title="Select Image Sequence (initial setting)")
    pass
    print("Image sequence initialized.")
    pass
    seq.uigetfile2(title="Current Image Sequence (2nd run)")
