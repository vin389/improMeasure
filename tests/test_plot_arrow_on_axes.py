import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def plot_arrow_on_axes(ax, p1, p2, color):
    xs = [p1[0], p2[0]]
    ys = [p1[1], p2[1]]
    zs = [p1[2], p2[2]]
    ax.quiver(xs[0], ys[0], zs[0], xs[1]-xs[0], ys[1]-ys[0], zs[1]-zs[0], color=color)

    # Set limits of axes
    max_val = np.max([np.abs(p1), np.abs(p2)])
    ax.set_xlim([-max_val,max_val])
    ax.set_ylim([-max_val,max_val])
    ax.set_zlim([-max_val,max_val])

# Example usage:
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Arrow
p1 = (0, 0, 0)
p2 = (1, 1, 1)

plot_arrow_on_axes(ax,p1,p2,'red')

plt.show()