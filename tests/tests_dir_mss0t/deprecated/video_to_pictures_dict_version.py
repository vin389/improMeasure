"""
This script contains two main tools for extracting images from video:

1. **video_to_pictures(video_path) -> dict**
   - A backend Python function that extracts and saves frames from a video file.

2. **VideoToPicturesGUI(tk.Frame)**
   - A Tkinter GUI frontend for user-friendly interaction with the function.

---

📌 **Function: video_to_pictures(inputs: dict) -> dict**

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

🧪 **Class: VideoToPicturesGUI**

A complete GUI that allows users to select video files and extract frames with options like:
- cropping,
- resizing,
- renaming formats,
- canceling the job,
- displaying estimated time.

---

🧪 **Some Example Uses**

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

# 7. GUI Usage
from video_to_pictures import VideoToPicturesGUI
root = tk.Tk()
VideoToPicturesGUI(root)
root.mainloop()

---
"""

# (Rest of the implementation would follow here)
import cv2
import os
import tkinter as tk
from tkinter import filedialog, messagebox
import time
from datetime import datetime
import threading

def video_to_pictures(inputs: dict) -> dict:
    video_path = inputs['video_path']
    frame_range = inputs['frame_range']
    crop_region = inputs['crop_region']
    output_size = inputs['output_size']
    filename_format, filename_start = inputs['filenames']
    status_callback = inputs.get('status_callback', None)
    cancel_callback = inputs.get('cancel_callback', lambda: False)

    x, y, w, h = crop_region
    start_frame, end_frame = frame_range

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    saved_files = []
    saved_indices = []
    total_frames = end_frame - start_frame + 1
    start_time = time.time()

    # Instead of using cap.set() to set the position to start_frame, we will read frames sequentially
    # from frame 0 to start_frame-1, then start saving frames from start_frame to end_frame.
    # 1. set the position to the very beginning of the video 
    # 2. read frames 0 to start_frame-1, and discard them
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for _ in range(start_frame):
            success, _ = cap.read()
            if not success:
                print(f"Warning: Cannot read frame {_}")
#    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame) # We do not trust cap.set(). 
    frame_idx = start_frame 

    for i in range(total_frames):
        if cancel_callback():
            break

        success, frame = cap.read()
        if not success:
            print(f"Warning: Cannot read frame {frame_idx}")
            frame_idx += 1
            continue

        cropped = frame[y:y+h, x:x+w]
        if output_size is not None:
            cropped = cv2.resize(cropped, output_size, interpolation=cv2.INTER_AREA)

        file_index = filename_start + len(saved_files)
        filename = filename_format % file_index
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        success = cv2.imwrite(filename, cropped)
        if success:
            saved_files.append(filename)
            saved_indices.append(frame_idx)

        if status_callback:
            elapsed = time.time() - start_time
            estimated_total = elapsed / (i + 1) * total_frames
            remaining = estimated_total - elapsed
            status_callback(i + 1, total_frames, elapsed, remaining)

        frame_idx += 1

    cap.release()

    return {
        'saved_files': saved_files,
        'frame_indices': saved_indices
    }

class VideoToPicturesGUI:
    def __init__(self, master):
        self.master = master
        master.title("Video to Pictures")

        self.inputs = {
            'video_path': tk.StringVar(),
            'frame_start': tk.IntVar(value=0),
            'frame_end': tk.IntVar(value=100),
            'crop_x': tk.IntVar(value=0),
            'crop_y': tk.IntVar(value=0),
            'crop_w': tk.IntVar(value=640),
            'crop_h': tk.IntVar(value=480),
            'resize_w': tk.IntVar(value=320),
            'resize_h': tk.IntVar(value=240),
            'filename_format': tk.StringVar(value="output/image_%05d.jpg"),
            'filename_start': tk.IntVar(value=0)
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

        tk.Label(self.master, text="Video Path:").grid(row=row, column=0, sticky='e')
        tk.Entry(self.master, textvariable=self.inputs['video_path'], width=50).grid(row=row, column=1, sticky='w')
        tk.Button(self.master, text="Browse", command=self.browse_video).grid(row=row, column=2)
        row += 1
        # add a read-only label to show video properties, including frame count, width, height, frame rate
        tk.Label(self.master, text="Video Properties:").grid(row=row, column=0, sticky='e')
        self.video_properties_label = tk.Label(self.master, text="(Select a video to see properties)")
        self.video_properties_label.grid(row=row, column=1, sticky='w')
        row += 1

        entry("Frame Start:", self.inputs['frame_start'])
        entry("Frame End:", self.inputs['frame_end'])
        entry("Crop X:", self.inputs['crop_x'])
        entry("Crop Y:", self.inputs['crop_y'])
        entry("Crop W:", self.inputs['crop_w'])
        entry("Crop H:", self.inputs['crop_h'])
        entry("Resize W:", self.inputs['resize_w'])
        entry("Resize H:", self.inputs['resize_h'])
        entry("Filename Format:", self.inputs['filename_format'], width=40)
        entry("Filename Start:", self.inputs['filename_start'])

        self.run_button = tk.Button(self.master, text="Run", command=self.run)
        self.run_button.grid(row=row, column=1)
        self.cancel_button = tk.Button(self.master, text="Cancel", command=self.cancel)
        self.cancel_button.grid(row=row, column=2)
        self.cancel_button.config(state='disabled')
        row += 1

        tk.Label(self.master, textvariable=self.status_text).grid(row=row, column=1, sticky='w')

    def browse_video(self):
        filename = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")])
        if filename:
            self.inputs['video_path'].set(filename)
            cap = cv2.VideoCapture(filename)
            if cap.isOpened():
                num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                self.inputs['frame_start'].set(0)
                self.inputs['frame_end'].set(num_frames - 1)
                self.inputs['crop_x'].set(0)
                self.inputs['crop_y'].set(0)
                self.inputs['crop_w'].set(width)
                self.inputs['crop_h'].set(height)
                self.inputs['resize_w'].set(width)
                self.inputs['resize_h'].set(height)
                cap.release()
                # update the video properties label. fps is rounded to 2 decimal places
                self.video_properties_label.config(text=f"Frames: {num_frames}, Width: {width}, Height: {height}, FPS: {round(fps, 2)}")

    def cancel(self):
        self.cancel_requested = True
        self.status_text.set("Cancelling...")

    def run(self):
        self.cancel_requested = False
        self.run_button.config(state='disabled')
        self.cancel_button.config(state='normal')

        def update_status(current, total, elapsed, remaining):
            self.status_text.set(f"{current}/{total} frames, elapsed: {elapsed:.1f}s, remaining: {remaining:.1f}s")

        def task():
            input_data = {
                'video_path': self.inputs['video_path'].get(),
                'frame_range': (self.inputs['frame_start'].get(), self.inputs['frame_end'].get()),
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
                'filenames': [
                    self.inputs['filename_format'].get(),
                    self.inputs['filename_start'].get()
                ],
                'status_callback': update_status,
                'cancel_callback': lambda: self.cancel_requested
            }

            try:
                video_to_pictures(input_data)
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
    app = VideoToPicturesGUI(root)
    root.mainloop()