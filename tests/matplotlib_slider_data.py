import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
# Create the figure and the line that we will manipulate
fig, ax = plt.subplots()
t = np.linspace(0, 1, 500)
initial_freq = 5
signal = np.sin(2 * np.pi * initial_freq * t)
line, = plt.plot(t, signal)
axfreq = plt.axes([0.25, 0.1, 0.65, 0.03])
freq_slider = Slider(axfreq, 'Frequency', 0.1, 30.0, valinit=initial_freq)
# Update the plot with the change from the slider
def update(val):
    line.set_ydata(np.sin(2 * np.pi * val * t))
    fig.canvas.draw_idle()
freq_slider.on_changed(update)
plt.show()