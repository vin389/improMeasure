import numpy as np

# This function creates 3D points along the 12 edges of a regular cube.
# The center of the cube is given (center), and the edge of the cube is given (edge). 
# The function returns a (12 x n_points_per_edge) x 3 numpy array.
# The returned array is reshaped from a 3D array, where the first dimension is the edge number (0-11).
# The second dimension is the number of points per edge (n_points_per_edge).
# The third dimension is the 3D position of the points.
# The points are uniformly distributed along the edge.
# The first point is at the center of the edge, and the last point is at the edge's end.
# The points are distributed in the order of the edge's direction.
# The edge's direction is along the x, y, or z axis.
# The edge number is as follows:
#   0: (center[0]-edge/2, center[1]-edge/2, center[2]-edge/2) to (center[0]+edge/2, center[1]-edge/2, center[2]-edge/2)
#   1: (center[0]+edge/2, center[1]-edge/2, center[2]-edge/2) to (center[0]+edge/2, center[1]+edge/2, center[2]-edge/2)
#   2: (center[0]+edge/2, center[1]+edge/2, center[2]-edge/2) to (center[0]-edge/2, center[1]+edge/2, center[2]-edge/2)
#   3: (center[0]-edge/2, center[1]+edge/2, center[2]-edge/2) to (center[0]-edge/2, center[1]-edge/2, center[2]-edge/2)
#   4: (center[0]-edge/2, center[1]-edge/2, center[2]+edge/2) to (center[0]+edge/2, center[1]-edge/2, center[2]+edge/2)
#   5: (center[0]+edge/2, center[1]-edge/2, center[2]+edge/2) to (center[0]+edge/2, center[1]+edge/2, center[2]+edge/2)
#   6: (center[0]+edge/2, center[1]+edge/2, center[2]+edge/2) to (center[0]-edge/2, center[1]+edge/2, center[2]+edge/2)
#   7: (center[0]-edge/2, center[1]+edge/2, center[2]+edge/2) to (center[0]-edge/2, center[1]-edge/2, center[2]+edge/2)
#   8: (center[0]-edge/2, center[1]-edge/2, center[2]-edge/2) to (center[0]-edge/2, center[1]-edge/2, center[2]+edge/2)
#   9: (center[0]+edge/2, center[1]-edge/2, center[2]-edge/2) to (center[0]+edge/2, center[1]-edge/2, center[2]+edge/2)
#  10: (center[0]+edge/2, center[1]+edge/2, center[2]-edge/2) to (center[0]+edge/2, center[1]+edge/2, center[2]+edge/2)
#  11: (center[0]-edge/2, center[1]+edge/2, center[2]-edge/2) to (center[0]-edge/2, center[1]+edge/2, center[2]+edge/2)
#  Input:
#    center: the center of the cube, represented by a tuple (xc, yc, zc), 
#            a list [xc, yc, zc], or a numpy vector np.array([xc, yc, zc]).
#            The default value is (0.0, 0.0, 0.0).
#    edge: the edge of the cube. The default is 1.0.
#    n_points_per_edge: the number of points per edge. The default is 5. Note that when you assign n_points_per_edge=5
#                    you see 6 points per edge, because the end point belongs to other edge. The points at 8 corners
#                    of this cube are not repeatedly generated. 
#  Output:
#    pos3ds: a 2D (12 x n_points_per_edge) x 3 numpy array. 
#            pos3ds is reshaped from a 12 by n_points_per_edge by 3 array, 
#            where the first dimension is the edge number (0-11).
#            The second dimension is the number of points per edge (n_points_per_edge).
#            The third dimension is the 3D position of the points.
#
# Example:
#    center = (0.0, 0.0, 0.0)
#    edge = 1.0
#    n_points_per_edge = 10
#    pos3ds = create_synthetic_3d_points_cube(center, edge, n_points_per_edge)
#    print(pos3ds)
#
def create_synthetic_3d_points_cube(center=(0.0, 0.0, 0.0), edge=1.0, n_points_per_edge=10):
    # create 3D points along the 12 edges of a regular cube
    # the 3D points are uniformly distributed along the edge
    # the first point is at the center of the edge, and the last point is at the edge's end
    # the points are distributed in the order of the edge's direction
    # the edge's direction is along the x, y, or z axis
    pos3ds = np.zeros((12,n_points_per_edge, 3))
    dx = edge / n_points_per_edge
    # edge 0
    pos3ds[0,:,0] = np.linspace(center[0]-edge/2, center[0]+edge/2 - dx, n_points_per_edge)
    pos3ds[0,:,1] = center[1]-edge/2
    pos3ds[0,:,2] = center[2]-edge/2
    # edge 1
    pos3ds[1,:,0] = center[0]+edge/2
    pos3ds[1,:,1] = np.linspace(center[1]-edge/2, center[1]+edge/2 - dx, n_points_per_edge)
    pos3ds[1,:,2] = center[2]-edge/2
    # edge 2
    pos3ds[2,:,0] = np.linspace(center[0]+edge/2, center[0]-edge/2 + dx, n_points_per_edge)
    pos3ds[2,:,1] = center[1]+edge/2
    pos3ds[2,:,2] = center[2]-edge/2
    # edge 3
    pos3ds[3,:,0] = center[0]-edge/2
    pos3ds[3,:,1] = np.linspace(center[1]+edge/2, center[1]-edge/2 + dx, n_points_per_edge)
    pos3ds[3,:,2] = center[2]-edge/2
    # edge 4
    pos3ds[4,:,0] = np.linspace(center[0]-edge/2, center[0]+edge/2 - dx, n_points_per_edge)
    pos3ds[4,:,1] = center[1]-edge/2
    pos3ds[4,:,2] = center[2]+edge/2
    # edge 5
    pos3ds[5,:,0] = center[0]+edge/2
    pos3ds[5,:,1] = np.linspace(center[1]-edge/2, center[1]+edge/2 - dx, n_points_per_edge)
    pos3ds[5,:,2] = center[2]+edge/2
    # edge 6
    pos3ds[6,:,0] = np.linspace(center[0]+edge/2, center[0]-edge/2 + dx, n_points_per_edge)
    pos3ds[6,:,1] = center[1]+edge/2
    pos3ds[6,:,2] = center[2]+edge/2
    # edge 7
    pos3ds[7,:,0] = center[0]-edge/2
    pos3ds[7,:,1] = np.linspace(center[1]+edge/2, center[1]-edge/2 + dx, n_points_per_edge)
    pos3ds[7,:,2] = center[2]+edge/2
    # edge 8
    pos3ds[8,:,0] = center[0]-edge/2
    pos3ds[8,:,1] = center[1]-edge/2
    pos3ds[8,:,2] = np.linspace(center[2]-edge/2, center[2]+edge/2 - dx, n_points_per_edge)
    # edge 9
    pos3ds[9,:,0] = center[0]+edge/2
    pos3ds[9,:,1] = center[1]-edge/2
    pos3ds[9,:,2] = np.linspace(center[2]-edge/2, center[2]+edge/2 - dx, n_points_per_edge)
    # edge 10
    pos3ds[10,:,0] = center[0]+edge/2
    pos3ds[10,:,1] = center[1]+edge/2
    pos3ds[10,:,2] = np.linspace(center[2]-edge/2, center[2]+edge/2 - dx, n_points_per_edge)
    # edge 11
    pos3ds[11,:,0] = center[0]-edge/2
    pos3ds[11,:,1] = center[1]+edge/2
    pos3ds[11,:,2] = np.linspace(center[2]-edge/2, center[2]+edge/2 - dx, n_points_per_edge)
    # reshape the array
    pos3ds = pos3ds.reshape(-1, 3)
    return pos3ds

# This function plots 3D points in a matplotlib axes
# If the given axes is None, a new figure will be created.
# If the given axes is not None, the plot will be added to the existing figure.
# This function returns the axes of the plot.
# Input:
#   pos3ds: a nPoints x 3 numpy array, where nPoints is the number of points.
#           Each row of pos3ds is a 3D point.
#   axes: the axes of the plot. The default is None.
#         If None, a new figure will be created.
#   consistentScale: make the scale consistent. The default is False.
# Output: 
#   axes: the axes of the plot.
def plot_3d_points(pos3ds, axes=None, consistentScale=True):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    # create a new figure if theAxes is None
    if axes is None:
        fig = plt.figure()
        axes = fig.add_subplot(111, projection='3d')
    ax = axes
    # plot the 3D points
    ax.scatter(pos3ds[:, 0], pos3ds[:, 1], pos3ds[:, 2])
    # label the axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.autoscale(enable=True, axis='both', tight=True)
#    plt.show(block=False)
    # make scale consistent
    if consistentScale == True:
        # Get current limits
        xlim = ax.get_xlim3d()
        ylim = ax.get_ylim3d()
        zlim = ax.get_zlim3d()
        # Calculate ranges
        xrange = abs(xlim[1] - xlim[0])
        yrange = abs(ylim[1] - ylim[0])
        zrange = abs(zlim[1] - zlim[0])
        # Find max range
        max_range = max([xrange, yrange, zrange])
        # Set new limits
        ax.set_xlim3d([np.mean(xlim) - max_range / 2, np.mean(xlim) + max_range / 2])
        ax.set_ylim3d([np.mean(ylim) - max_range / 2, np.mean(ylim) + max_range / 2])
        ax.set_zlim3d([np.mean(zlim) - max_range / 2, np.mean(zlim) + max_range / 2])
    return ax

# test
if __name__ == '__main__':
    # create 3D points along the 12 edges of a regular cube
    center1 = (0.0, 0.0, 0.0)
    edge1 = 1.0
    n_points_per_edge1 = 5
    pos3ds1 = create_synthetic_3d_points_cube(center1, edge1, n_points_per_edge1)
    print(pos3ds1)
    # plot the 3D points
    # create a figure with a 3d axes
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plot_3d_points(pos3ds1, ax, consistentScale=False)

    # create 3D points along the 12 edges of a regular cube
    center2 = (3.0, 0.0, 0.0)
    edge2 = 1.0
    n_points_per_edge2 = 5
    pos3ds2 = create_synthetic_3d_points_cube(center2, edge2, n_points_per_edge2)
    print(pos3ds2)
    # plot the 3D points
    # create a figure with a 3d axes
    plot_3d_points(pos3ds2, ax, consistentScale=True)

    # Make scale consistent

    plt.show(block=True)




