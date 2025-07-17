"""
This script contains two tools for extracting images from video:

1. **video_to_pictures(...)**
   - A Python function that extracts and saves frames from a video file.

2. **VideoToPicturesGUI(tk.Frame)**
   - A Tkinter GUI frontend for user-friendly interaction.

---

**Function: video_to_pictures(...)**

Arguments:
- `video_path`: Path to the input video file.
- `frame_range`: (start, end) tuple for which frames to extract.
- `crop_region`: (x, y, w, h) crop rectangle.
- `output_size`: (w, h) size to resize the cropped image.
- `filenames`: (format_str, start_index) tuple like ('img_%05d.jpg', 0).
- `status_callback`: optional function to update progress.
- `cancel_callback`: optional function to cancel early.

Returns:
- `saved_files`: List of saved file paths.
- `frame_indices`: List of frame indices extracted.

---

Examples:

video_to_pictures("video.mp4", (0,100), (0,0,640,480), (320,240), ("img_%04d.jpg",0))
"""

import cv2
import os
import tkinter as tk
from tkinter import filedialog, messagebox
import time
import threading

def video_to_pictures(
    video_path,
    frame_range,
    crop_region,
    output_size,
    filenames,
    status_callback=None,
    cancel_callback=lambda: False
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("Cannot open video file")

    x, y, w, h = crop_region
    resize_w, resize_h = output_size
    fmt_str, start_idx = filenames

    saved_files = []
    frame_indices = []

    total_frames = frame_range[1] - frame_range[0] + 1
    frame_idx = 0
    saved_idx = start_idx

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_range[0])
    start_time = time.time()

    for i in range(total_frames):
        if cancel_callback():
            break

        success, frame = cap.read()
        if not success:
            break

        cropped = frame[y:y+h, x:x+w]
        resized = cv2.resize(cropped, (resize_w, resize_h), interpolation=cv2.INTER_AREA)
        fname = fmt_str % saved_idx
        cv2.imwrite(fname, resized)
        saved_files.append(fname)
        frame_indices.append(frame_range[0] + i)
        saved_idx += 1

        if status_callback:
            elapsed = time.time() - start_time
            eta = elapsed / (i + 1) * total_frames - elapsed
            status_callback(i + 1, total_frames, elapsed, eta)

    cap.release()
    return saved_files, frame_indices

class VideoToPicturesGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Video to Pictures")
        self.inputs = {}
        self.cancel_flag = False

        row = 0
        tk.Label(master, text="Video file").grid(row=row, column=0, sticky='w')
        self.inputs['video_path'] = tk.StringVar()
        tk.Entry(master, textvariable=self.inputs['video_path'], width=60).grid(row=row, column=1, sticky='w')
        tk.Button(master, text="Browse...", command=self.browse_video).grid(row=row, column=2)

        for label in ['frame_start', 'frame_end', 'crop_x', 'crop_y', 'crop_w', 'crop_h', 'resize_w', 'resize_h', 'filename_format', 'filename_start']:
            row += 1
            tk.Label(master, text=label).grid(row=row, column=0, sticky='w')
            self.inputs[label] = tk.StringVar()
            tk.Entry(master, textvariable=self.inputs[label], width=30).grid(row=row, column=1, sticky='w')

        self.inputs['filename_format'].set("output_%05d.jpg")
        self.inputs['filename_start'].set("0")

        row += 1
        self.status_var = tk.StringVar()
        tk.Label(master, textvariable=self.status_var).grid(row=row, column=0, columnspan=3, sticky='w')

        row += 1
        self.run_button = tk.Button(master, text="Run", command=self.run)
        self.run_button.grid(row=row, column=0, sticky='w')
        self.cancel_button = tk.Button(master, text="Cancel running", command=self.cancel)
        self.cancel_button.grid(row=row, column=1, sticky='w')
        self.cancel_button.config(state='disabled')

    def browse_video(self):
        filename = filedialog.askopenfilename(title="Select video file")
        if filename:
            self.inputs['video_path'].set(filename)
            cap = cv2.VideoCapture(filename)
            if cap.isOpened():
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
                self.inputs['frame_start'].set("0")
                self.inputs['frame_end'].set(str(frame_count - 1))
                self.inputs['crop_x'].set("0")
                self.inputs['crop_y'].set("0")
                self.inputs['crop_w'].set(str(width))
                self.inputs['crop_h'].set(str(height))
                self.inputs['resize_w'].set(str(width))
                self.inputs['resize_h'].set(str(height))

    def run(self):
        self.cancel_flag = False
        self.run_button.config(state='disabled')
        self.cancel_button.config(state='normal')
        threading.Thread(target=self.task).start()

    def cancel(self):
        self.cancel_flag = True

    def task(self):
        try:
            video_path = self.inputs['video_path'].get()
            frame_start = int(self.inputs['frame_start'].get())
            frame_end = int(self.inputs['frame_end'].get())
            crop_x = int(self.inputs['crop_x'].get())
            crop_y = int(self.inputs['crop_y'].get())
            crop_w = int(self.inputs['crop_w'].get())
            crop_h = int(self.inputs['crop_h'].get())
            resize_w = int(self.inputs['resize_w'].get())
            resize_h = int(self.inputs['resize_h'].get())
            filename_format = self.inputs['filename_format'].get()
            filename_start = int(self.inputs['filename_start'].get())

            def status(cur, total, elapsed, eta):
                self.status_var.set(f"{cur}/{total} | Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s")

            def cancel():
                return self.cancel_flag

            video_to_pictures(
                video_path=video_path,
                frame_range=(frame_start, frame_end),
                crop_region=(crop_x, crop_y, crop_w, crop_h),
                output_size=(resize_w, resize_h),
                filenames=(filename_format, filename_start),
                status_callback=status,
                cancel_callback=cancel
            )

            self.status_var.set("Done.")
        except Exception as e:
            self.status_var.set(f"Error: {e}")
        finally:
            self.run_button.config(state='normal')
            self.cancel_button.config(state='disabled')

if __name__ == "__main__":
    root = tk.Tk()
    VideoToPicturesGUI(root)
    root.mainloop()
