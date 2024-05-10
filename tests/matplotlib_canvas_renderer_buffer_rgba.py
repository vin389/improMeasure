import cv2
import matplotlib.pyplot as plt
import numpy as np

# Sample data (replace with your actual data)
x = np.linspace(0.0, 5.0, 100)  # 100 points between 0 and 5
y = np.sin(x)  # Example function

# Create the plot
plt.ioff()
plt.rcParams['figure.dpi'] = 100 # 100 DPI
fig, ax = plt.subplots(figsize=(6, 4))  # figsize in inch
ax.plot(x, y)

# Add labels and title (optional)
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_title("Sample Curve")

# Force the canvas to draw (important for conversion)
fig.canvas.draw()

# Get the data from the canvas as RGB image
img_plot = np.array(fig.canvas.renderer.buffer_rgba())

# Convert the image format from RGBA to BGR (OpenCV default)
img_plot = cv2.cvtColor(img_plot, cv2.COLOR_RGBA2BGR)
print("Dimension of canvas:", img_plot.shape)
cv2.imshow("Matplotlib canvas renderer buffer_rgba", img_plot)
# You can now use the img_plot as an OpenCV image for further processing
# ...

# Optional: Close the plot window (if one is created internally)
#plt.close()
