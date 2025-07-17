"""
This script contains two main tools for extracting images from video:

1. **video_to_pictures(inputs: dict) -> dict**
   - A backend Python function that extracts and saves frames from a video file.

2. **VideoToPicturesGUI(tk.Frame)**
   - A Tkinter GUI frontend for user-friendly interaction with the function.

---

**Function: video_to_pictures(inputs: dict) -> dict**

This function accepts a dictionary of parameters:
- `video_path`: Full path to the input video file.
- `frame_range`: Tuple (start_frame, end_frame) indicating which frames to extract.
- `crop_region`: Tuple (x, y, w, h) specifying a rectangular region to crop.
- `output_size`: Tuple (w, h) for the resized image.
- `filenames`: A tuple of (format_string, start_index), e.g. ("output/img_%05d.jpg", 0).
- `status_callback`: Optional function that updates progress.
- `cancel_callback`: Optional function that allows interruption.

The function returns a dictionary:
- `saved_files`: List of file paths saved.
- `frame_indices`: List of frame indices that were successfully processed.

---

**Class: VideoToPicturesGUI**

A complete GUI that allows users to select video files and extract frames with options like:
- cropping,
- resizing,
- renaming formats,
- canceling the job,
- displaying estimated time.

---

**10 Example Uses**

```python
# 1. Basic video to images
video_to_pictures({
    'video_path': 'video.mp4',
    'frame_range': (0, 100),
    'crop_region': (0, 0, 1920, 1080),
    'output_size': (640, 360),
    'filenames': ['output/frame_%05d.jpg', 0]
})

# 2. Crop center square
video_to_pictures({
    'video_path': 'input.avi',
    'frame_range': (50, 60),
    'crop_region': (100, 100, 500, 500),
    'output_size': (250, 250),
    'filenames': ['data/square_%03d.png', 10]
})

# 3. No resizing (same size)
video_to_pictures({
    'video_path': 'cam.mov',
    'frame_range': (0, 5),
    'crop_region': (0, 0, 640, 480),
    'output_size': (640, 480),
    'filenames': ['pics/pic_%d.jpg', 1]
})

# 4. Save as PNG
video_to_pictures({
    'video_path': 'clip.mp4',
    'frame_range': (10, 20),
    'crop_region': (0, 0, 1280, 720),
    'output_size': (640, 360),
    'filenames': ['snapshots/img_%04d.png', 100]
})

# 5. Track progress in terminal
video_to_pictures({
    'video_path': 'footage.mp4',
    'frame_range': (0, 50),
    'crop_region': (0,0,800,600),
    'output_size': (400,300),
    'filenames': ['frames/f_%03d.jpg', 0],
    'status_callback': lambda cur, total, e, r: print(f"{cur}/{total} Elapsed:{e:.1f}s Remaining:{r:.1f}s")
})

# 6. Cancel halfway
cancel_flag = False
def should_cancel(): return cancel_flag
video_to_pictures({
    'video_path': 'test.mp4',
    'frame_range': (0, 999),
    'crop_region': (0,0,640,480),
    'output_size': (320,240),
    'filenames': ['out/img_%03d.jpg', 0],
    'cancel_callback': should_cancel
})

# 7. Large video export, resized
...

# 8. Small time-lapse crop
...

# 9. GUI Usage
from video_to_pictures import VideoToPicturesGUI
root = tk.Tk()
VideoToPicturesGUI(root)
root.mainloop()

# 10. Called in batch script
for i in range(5):
    video_to_pictures({...})
```
---
"""

import cv2
import os
import tkinter as tk
from tkinter import filedialog, messagebox
import time
from datetime import datetime
import threading

def pictures_to_video(inputs: dict) -> dict:
    filenames = inputs['filenames']  # list of image file paths
    crop_region = inputs['crop_region']  # (x, y, w, h)
    output_size = inputs['output_size']  # (w, h)
    codec = inputs['codec']
    fps = inputs['fps']
    output_file = inputs['output_file']
    status_callback = inputs.get('status_callback', None)
    cancel_callback = inputs.get('cancel_callback', lambda: False)

    if not filenames:
        raise ValueError("No input image files provided.")

    x, y, w, h = crop_region

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(output_file, fourcc, fps, output_size)

    start_time = time.time()
    total_frames = len(filenames)

    for i, file in enumerate(filenames):
        if cancel_callback():
            break

        img = cv2.imread(file)
        if img is None:
            print(f"Warning: Cannot read image {file}")
            continue

        cropped = img[y:y+h, x:x+w]
        if output_size is not None:
            cropped = cv2.resize(cropped, output_size, interpolation=cv2.INTER_AREA)

        out.write(cropped)

        if status_callback:
            elapsed = time.time() - start_time
            estimated_total = elapsed / (i + 1) * total_frames
            remaining = estimated_total - elapsed
            status_callback(i + 1, total_frames, elapsed, remaining)

    out.release()

    return {
        'output_file': output_file,
        'frame_count': total_frames
    }

class PicturesToVideoGUI:
    def __init__(self, master):
        self.master = master
        master.title("Pictures to Video")

        self.inputs = {
            'filenames': [],
            'crop_x': tk.IntVar(value=0),
            'crop_y': tk.IntVar(value=0),
            'crop_w': tk.IntVar(value=640),
            'crop_h': tk.IntVar(value=480),
            'resize_w': tk.IntVar(value=640),
            'resize_h': tk.IntVar(value=480),
            'codec': tk.StringVar(value='mp4v'),
            'fps': tk.DoubleVar(value=30.0),
            'output_file': tk.StringVar(value='')
        }

        self.status_text = tk.StringVar(value="")
        self.cancel_requested = False
        self.run_button = None
        self.create_widgets()

    def create_widgets(self):
        row = 0
        def entry(label, var, width=10):
            nonlocal row
            tk.Label(self.master, text=label).grid(row=row, column=0, sticky='e')
            tk.Entry(self.master, textvariable=var, width=width).grid(row=row, column=1, sticky='w')
            row += 1

        tk.Button(self.master, text="Browse Images", command=self.browse_images).grid(row=row, column=0, columnspan=2)
        row += 1

        # label to show information about the selected images. Resolution, number of images, 
        # first file name, last file name. 
        self.image_info_label = tk.Label(self.master, text="No images selected.")
        self.image_info_label.grid(row=row, column=0, columnspan=3, sticky='w')
        row += 1

        entry("Crop X:", self.inputs['crop_x'])
        entry("Crop Y:", self.inputs['crop_y'])
        entry("Crop W:", self.inputs['crop_w'])
        entry("Crop H:", self.inputs['crop_h'])
        entry("Resize W:", self.inputs['resize_w'])
        entry("Resize H:", self.inputs['resize_h'])
        entry("Codec:", self.inputs['codec'])
        entry("FPS:", self.inputs['fps'])
        entry("Output File:", self.inputs['output_file'], width=40)

        tk.Button(self.master, text="Browse Output", command=self.browse_output_file).grid(row=row-1, column=2)

        self.run_button = tk.Button(self.master, text="Run", command=self.run)
        self.run_button.grid(row=row, column=1)
        self.cancel_button = tk.Button(self.master, text="Cancel", command=self.cancel)
        self.cancel_button.grid(row=row, column=2)
        self.cancel_button.config(state='disabled')

        row += 1

        tk.Label(self.master, textvariable=self.status_text).grid(row=row, column=1, sticky='w')

    def browse_images(self):
        files = filedialog.askopenfilenames(filetypes=[("Image files", "*.jpg *.png *.bmp")])
        if files:
            self.inputs['filenames'] = list(files)
            first_file = files[0]
            img = cv2.imread(first_file)
            if img is not None:
                h, w = img.shape[:2]
                self.inputs['crop_x'].set(0)
                self.inputs['crop_y'].set(0)
                self.inputs['crop_w'].set(w)
                self.inputs['crop_h'].set(h)
                self.inputs['resize_w'].set(w)
                self.inputs['resize_h'].set(h)
                # update the image info label
                self.image_info_label.config(text=f"Selected {len(files)} images, "
                    f"first: {os.path.basename(first_file)}, " 
                    f"last: {os.path.basename(files[-1])}, " 
                    f"resolution: {w}x{h}")

            now = datetime.now().strftime("%Y%m%d_%H%M%S")
            dir_path = os.path.dirname(first_file)
            filelist_path = os.path.join(dir_path, f"filelist_{now}.txt")
            with open(filelist_path, 'w') as f:
                f.write('\n'.join(files))
            messagebox.showinfo("File List Saved", f"File list saved to {filelist_path}")

            output_file = os.path.join(dir_path, f"video_{now}.mp4")
            self.inputs['output_file'].set(output_file)

    def browse_output_file(self):
        file = filedialog.asksaveasfilename(defaultextension=".mp4",
                                             filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")])
        if file:
            self.inputs['output_file'].set(file)

    def cancel(self):
        self.cancel_requested = True
        self.status_text.set("Cancelling...")

    def run(self):
        self.cancel_requested = False
        self.run_button.config(state='disabled')
        self.cancel_button.config(state='normal')

        def update_status(current, total, elapsed, remaining):
            self.status_text.set(f"{current}/{total} images, elapsed: {elapsed:.1f}s, remaining: {remaining:.1f}s")

        def task():
            try:
                pictures_to_video({
                    'filenames': self.inputs['filenames'],
                    'crop_region': (
                        self.inputs['crop_x'].get(),
                        self.inputs['crop_y'].get(),
                        self.inputs['crop_w'].get(),
                        self.inputs['crop_h'].get()
                    ),
                    'output_size': (
                        self.inputs['resize_w'].get(),
                        self.inputs['resize_h'].get()
                    ),
                    'codec': self.inputs['codec'].get(),
                    'fps': self.inputs['fps'].get(),
                    'output_file': self.inputs['output_file'].get(),
                    'status_callback': update_status,
                    'cancel_callback': lambda: self.cancel_requested
                })
                if not self.cancel_requested:
                    self.status_text.set("Done.")
            except Exception as e:
                messagebox.showerror("Error", str(e))
            finally:
                self.run_button.config(state='normal')
                self.cancel_button.config(state='disabled')

        threading.Thread(target=task).start()

if __name__ == "__main__":
    root = tk.Tk()
    app = PicturesToVideoGUI(root)
    root.mainloop()
