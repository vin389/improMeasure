import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# Define the function for the sine curve
def f(x):
  return np.sin(2*np.pi*x)

# Initialize data (small for initial plot)
x_data = np.linspace(0, 1, 100)  # Adjust range for desired period
y_data = f(x_data)

# Create the plot
fig, ax = plt.subplots()
line, = ax.plot(x_data, y_data, marker='.', markersize=5, linewidth=0, label='Sine Curve')

# Set plot limits (adjustable based on your function)
ax.set_xlim(0, 2)  # Adjust for desired number of cycles
ax.set_ylim(-1.2, 1.2)

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Animated Sine Curve')


# Function to update data each frame
def animate(i):
  #global x_data, y_data
  # Update x data by shifting existing data and adding new point
  i = i % 200
  x_data = np.linspace(i, i+100, 101) % 200
  x_data /= 100
#  x_data = x_data[1:]
#  x_data = np.append(x_data, x_data[-1] + 0.1)  # Adjust new point increment
#  print(len(x_data))
  # Update y data based on the new x data
  y_data = f(x_data)

  # Update line data
  line.set_xdata(x_data)
  line.set_ydata(y_data)

  # Return updated artist
  return line,

# Animate the plot
animation = FuncAnimation(fig, animate, frames=400, interval=1)

plt.legend()
plt.show()
