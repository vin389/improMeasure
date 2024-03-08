import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def plot_triangle_on_figure(ax, p1, p2, p3, color):
    xs = [p1[0], p2[0], p3[0]]
    ys = [p1[1], p2[1], p3[1]]
    zs = [p1[2], p2[2], p3[2]]
    ax.plot_trisurf(xs, ys, zs, color=color)

# Example usage:
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# First triangle
p1 = (0, 0, 0)
p2 = (1, 0, 0)
p3 = (0.5, 1, 0)

plot_triangle_on_figure(ax, p1, p2, p3, 'red')

# Second triangle
p4 = (0.5, 0.5, 1)

plot_triangle_on_figure(ax,p4,p1,p2,'blue')

plt.show()