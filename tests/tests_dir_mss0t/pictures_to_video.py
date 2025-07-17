"""
This script contains two tools for combining pictures into a video:

1. **pictures_to_video(...)**
   - A backend Python function that creates a video from image files.

2. **PicturesToVideoGUI(tk.Frame)**
   - A Tkinter GUI frontend for user-friendly interaction with the function.

---

**Function: pictures_to_video(...)**

This function accepts conventional Python arguments:
- `filenames`: List of image file paths.
- `crop_region`: Tuple (x, y, w, h) specifying the crop rectangle.
- `output_size`: Tuple (w, h) for the resized output image.
- `codec`: Four-character code (e.g., 'mp4v').
- `fps`: Frame rate (float).
- `output_file`: Path of the video output file.
- `status_callback`: Optional function to report progress.
- `cancel_callback`: Optional function to support cancellation.

Returns:
- `saved_file`: Path to the generated video.
- `frame_count`: Number of frames written.

---

**10 Example Uses**

```python
# 1. Minimal usage
pictures_to_video(filenames=['img1.jpg', 'img2.jpg'], crop_region=(0,0,640,480),
                  output_size=(320,240), codec='mp4v', fps=24, output_file='out.mp4')

# 2. Skip progress and cancel callback
pictures_to_video([...], ..., status_callback=None, cancel_callback=None)

# 3. Track progress
pictures_to_video(..., status_callback=lambda cur, tot, e, r: print(cur, '/', tot))

# 4. Cancel mid-process
flag = False
def cancel(): return flag
pictures_to_video(..., cancel_callback=cancel)

# 5. Large video
pictures_to_video([...1000 images...])

# 6. Custom codec and fps
pictures_to_video(..., codec='XVID', fps=60.0)

# 7. High-res export
pictures_to_video(..., output_size=(1920, 1080))

# 8. GUI mode
from pictures_to_video import PicturesToVideoGUI
root = tk.Tk()
PicturesToVideoGUI(root)
root.mainloop()

# 9. Batch conversion
for group in groups:
    pictures_to_video(...)

# 10. File-based progress logging
with open('log.txt','a') as f:
    pictures_to_video(..., status_callback=lambda c,t,e,r: f.write(f"{c}/{t}\n"))
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

def pictures_to_video(
    filenames,
    crop_region,
    output_size,
    codec,
    fps,
    output_file,
    status_callback=None,
    cancel_callback=lambda: False
):
    if not filenames:
        raise ValueError("No input image files provided.")

    x, y, w, h = crop_region
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(output_file, fourcc, fps, output_size)

    start_time = time.time()
    total_frames = len(filenames)

    count_saved_frames = 0
    img_resolution = None
    for i, file in enumerate(filenames):
        if cancel_callback():
            break

        img = cv2.imread(file)
        if img is None:
            print(f"Warning: Cannot read image {file}")
            continue
        if img_resolution is None:
            img_resolution = img.shape[:2]
        if img_resolution != img.shape[:2]:
            print(f"Warning: Image {file} has different resolution {img.shape[:2]} than first image {img_resolution}. Skipping.")
            continue

        cropped = img[y:y+h, x:x+w]
        if output_size is not None:
            cropped = cv2.resize(cropped, output_size, interpolation=cv2.INTER_AREA)

        out.write(cropped)
        count_saved_frames += 1

        if status_callback:
            elapsed = time.time() - start_time
            estimated_total = elapsed / (i + 1) * total_frames
            remaining = estimated_total - elapsed
            status_callback(i + 1, total_frames, elapsed, remaining)

    out.release()

    return output_file, count_saved_frames

class PicturesToVideoGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Pictures to Video")
        self.inputs = {}
        self.cancel_flag = False
        self.return_value = None

        row = 0
        tk.Label(master, text="Image files").grid(row=row, column=0, sticky='w')
        self.inputs['filenames'] = tk.StringVar()
        tk.Entry(master, textvariable=self.inputs['filenames'], width=60).grid(row=row, column=1, sticky='w')
        tk.Button(master, text="Browse...", command=self.browse_files).grid(row=row, column=2, sticky='w')

        row += 1
        # a label that displays the number of selected files and the resolution of the first image
        self.file_info_label = tk.Label(master, text="No files selected.")
        self.file_info_label.grid(row=row, column=0, columnspan=3, sticky='w')

        row += 1
        tk.Label(master, text="Output file").grid(row=row, column=0, sticky='w')
        self.inputs['output_file'] = tk.StringVar()
        tk.Entry(master, textvariable=self.inputs['output_file'], width=60).grid(row=row, column=1, sticky='w')
        tk.Button(master, text="Browse...", command=self.browse_output).grid(row=row, column=2, sticky='w')

        fields = ['crop_x', 'crop_y', 'crop_w', 'crop_h', 'resize_w', 'resize_h', 'codec', 'fps']
        defaults = [0, 0, 640, 480, 640, 480, 'mp4v', 24.0]
        for i, field in enumerate(fields):
            row += 1
            tk.Label(master, text=field).grid(row=row, column=0, sticky='w')
            self.inputs[field] = tk.StringVar(value=str(defaults[i]))
            tk.Entry(master, textvariable=self.inputs[field], width=20).grid(row=row, column=1, sticky='w')

        row += 1
        self.status_var = tk.StringVar()
        tk.Label(master, textvariable=self.status_var).grid(row=row, column=0, columnspan=3, sticky='w')

        row += 1
        self.run_button = tk.Button(master, text="Run", command=self.run)
        self.run_button.grid(row=row, column=0, sticky='w')
        self.cancel_button = tk.Button(master, text="Cancel running", command=self.cancel)
        self.cancel_button.grid(row=row, column=1, sticky='w')
        self.cancel_button.config(state='disabled')

    def browse_files(self):
        files = filedialog.askopenfilenames(title="Select image files")
        if files:
            self.inputs['filenames'].set(';'.join(files))
            first = cv2.imread(files[0])
            if first is not None:
                h, w = first.shape[:2]
                self.inputs['crop_w'].set(w)
                self.inputs['crop_h'].set(h)
                self.inputs['resize_w'].set(w)
                self.inputs['resize_h'].set(h)
                dt = datetime.now().strftime("%Y%m%d_%H%M%S")
                video_path = os.path.join(os.path.dirname(files[0]), f"video_{dt}.mp4")
                self.inputs['output_file'].set(video_path)
                # update file info label
                self.file_info_label.config(text=f"{len(files)} files selected. Image resolution: {w}x{h}")

    def browse_output(self):
        filename = filedialog.asksaveasfilename(defaultextension=".mp4", title="Select output video file")
        if filename:
            self.inputs['output_file'].set(filename)

    def cancel(self):
        self.cancel_flag = True

    def run(self):
        self.cancel_flag = False
        self.run_button.config(state='disabled')
        self.cancel_button.config(state='normal')
        threading.Thread(target=self.task).start()

    def task(self):
        try:
            filenames = self.inputs['filenames'].get().split(';')
            x = int(self.inputs['crop_x'].get())
            y = int(self.inputs['crop_y'].get())
            w = int(self.inputs['crop_w'].get())
            h = int(self.inputs['crop_h'].get())
            rw = int(self.inputs['resize_w'].get())
            rh = int(self.inputs['resize_h'].get())
            codec = self.inputs['codec'].get()
            fps = float(self.inputs['fps'].get())
            output_file = self.inputs['output_file'].get()

            def status(cur, total, elapsed, remaining):
                self.status_var.set(f"{cur}/{total} frames | Elapsed: {elapsed:.1f}s | Remaining: {remaining:.1f}s")

            def cancel():
                return self.cancel_flag

            self.return_value =\
            pictures_to_video(
                filenames=filenames,
                crop_region=(x, y, w, h),
                output_size=(rw, rh),
                codec=codec,
                fps=fps,
                output_file=output_file,
                status_callback=status,
                cancel_callback=cancel
            )
            if self.return_value is None or self.return_value[1] != len(filenames):
                self.status_var.set("Not done. Only %d frames of %d files are saved." % (self.return_value[1], len(filenames)))
            else:
                self.status_var.set("Done. %d frames saved." % (self.return_value[1]))
        except Exception as e:
            self.status_var.set(f"Error: {e}")
        finally:
            self.run_button.config(state='normal')
            self.cancel_button.config(state='disabled')

if __name__ == '__main__':
    root = tk.Tk()
    p2v = PicturesToVideoGUI(root)
    root.mainloop()
    print("Return value:", p2v.return_value)
    if p2v.return_value is None:
        print("No video file was created.")
    else:
        # p2v.return_value is a tuple (output_file, frame_count)
        print("Main program: %d pictures are saved to video file:%s" % (p2v.return_value[1], p2v.return_value[0]))
