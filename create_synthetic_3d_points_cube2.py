import numpy as np
# This function creates 3D points along the 12 edges of a regular cube.
# The center of the cube is given (center), and the edges of the cube are given (edges). 
# The center is a tuple or list of three numbers (xc, yc, zc).
# The edges is a tuple or list of three numbers (x_edge, y_edge, z_edge).
# The npoints_per_edge is the number of points per edge (np_x, np_y, np_z).
# The function returns a n_points x 3 numpy array, where n_points is the total number of points.
# The points are uniformly distributed along the edge.
# Points at both ends of an edge could be duplicated because they are shared by three edges. 
# Example 1:
#   pos3Ds = create_synthetic_3d_points_cube2(
#       center=(0.0, 0.0, 0.0), 
#       edges=(1.0, 1.5, 2.0), 
#       npoints_per_edge=(5, 7, 9))
# Example 2:
#   pos3Ds = create_synthetic_3d_points_cube2(
#       center=(0.0, 0.0, 2.0), 
#       edges=(5.4, 5.4, 2.4), 
#       npoints_per_edge=(10, 10, 3+))
def create_synthetic_3d_points_cube2(center=(0.0, 0.0, 0.0), edges=(1.0, 1.0, 1.0), npoints_per_edge=(10, 10, 10)):
    # if edges is a single value, convert it to a tuple of three values
    if isinstance(edges, (int, float)):
        edges = (edges, edges, edges)
    # if npoints_per_edge is a single value, convert it to a tuple of three values
    if isinstance(npoints_per_edge, int):
        npoints_per_edge = (npoints_per_edge, npoints_per_edge, npoints_per_edge)
    # create 3D points along the 12 edges of a regular cube
    n_points = 4 * (npoints_per_edge[0] + npoints_per_edge[1] + npoints_per_edge[2])
    pos3Ds = np.zeros((n_points, 3))
    # generate points along the x-axis
    x = np.linspace(-edges[0] / 2, edges[0] / 2, npoints_per_edge[0])
    # generate points along the y-axis
    y = np.linspace(-edges[1] / 2, edges[1] / 2, npoints_per_edge[1])
    # generate points along the z-axis
    z = np.linspace(-edges[2] / 2, edges[2] / 2, npoints_per_edge[2])
    # generate points along the edges
    i = 0
    for j in range(npoints_per_edge[0]):
        pos3Ds[i, :] = np.array([x[j], -edges[1] / 2, -edges[2] / 2]) + np.array(center)
        i += 1
    for j in range(npoints_per_edge[0]):
        pos3Ds[i, :] = np.array([x[j], edges[1] / 2, -edges[2] / 2]) + np.array(center)
        i += 1
    for j in range(npoints_per_edge[0]):
        pos3Ds[i, :] = np.array([x[j], edges[1] / 2, edges[2] / 2]) + np.array(center)
        i += 1
    for j in range(npoints_per_edge[0]):
        pos3Ds[i, :] = np.array([x[j], -edges[1] / 2, edges[2] / 2]) + np.array(center)
        i += 1
    for j in range(npoints_per_edge[1]):
        pos3Ds[i, :] = np.array([-edges[0] / 2, y[j], -edges[2] / 2]) + np.array(center)
        i += 1
    for j in range(npoints_per_edge[1]):
        pos3Ds[i, :] = np.array([ edges[0] / 2, y[j], -edges[2] / 2]) + np.array(center)
        i += 1
    for j in range(npoints_per_edge[1]):
        pos3Ds[i, :] = np.array([ edges[0] / 2, y[j],  edges[2] / 2]) + np.array(center)
        i += 1
    for j in range(npoints_per_edge[1]):
        pos3Ds[i, :] = np.array([-edges[0] / 2, y[j],  edges[2] / 2]) + np.array(center)
        i += 1
    for j in range(npoints_per_edge[2]):
        pos3Ds[i, :] = np.array([-edges[0] / 2, -edges[1] / 2, z[j]]) + np.array(center)
        i += 1
    for j in range(npoints_per_edge[2]):
        pos3Ds[i, :] = np.array([ edges[0] / 2, -edges[1] / 2, z[j]]) + np.array(center)
        i += 1
    for j in range(npoints_per_edge[2]):
        pos3Ds[i, :] = np.array([ edges[0] / 2,  edges[1] / 2, z[j]]) + np.array(center)
        i += 1
    for j in range(npoints_per_edge[2]):
        pos3Ds[i, :] = np.array([-edges[0] / 2,  edges[1] / 2, z[j]]) + np.array(center)
        i += 1
    # remove duplicated points
    # pos3Ds = np.unique(pos3Ds, axis=0) # unique() is used by github copilot but I don't truss it because it does not allow me to set tolerance. sorry.
    pos3Ds_clone = np.zeros(pos3Ds.shape, dtype=float)
    count = 0
    for i in range(pos3Ds.shape[0]):
        point_i_is_unique = True
        for j in range(i+1, pos3Ds.shape[0]):
            if np.allclose(pos3Ds[i], pos3Ds[j], atol=1e-9):
                point_i_is_unique = False
                break
        if point_i_is_unique:
            pos3Ds_clone[count] = pos3Ds[i]
            count += 1
    pos3Ds = pos3Ds_clone[:count]
    return pos3Ds


# This function plots the 3D points in a 3D plot using matplotlib for user to verify the 3D points.
if __name__ == '__main__':
    # create a tk window with a size of 1200 x 500 pixels
    import tkinter as tk
    root = tk.Tk()
    root.geometry('1200x500')
    # create a label at left and a text entry widget at right for the center of the sphere
    irow = 0    
    label = tk.Label(root, text='Center of the cube:')
    label.grid(row=irow, column=0)
    tk_center = tk.Entry(root)
    tk_center.insert(0, '0.0, 0.0, 0.0')
    tk_center.grid(row=irow, column=1)
    tk_center.configure(width=60)
    # create a label at left and a text entry widget at right for the edges of the cube
    irow += 1
    label = tk.Label(root, text='Edges of the cube:')
    label.grid(row=irow, column=0)
    tk_edges = tk.Entry(root)
    tk_edges.insert(0, '1.0, 1.5, 2.0')
    tk_edges.grid(row=irow, column=1)
    tk_edges.configure(width=60)
    # create a label at left and a text entry widget at right for the numbers of points per edge
    irow += 1
    label = tk.Label(root, text='Number of points per edge:')
    label.grid(row=irow, column=0)
    tk_npoints_per_edge = tk.Entry(root)
    tk_npoints_per_edge.insert(0, '5, 7, 9')
    tk_npoints_per_edge.grid(row=irow, column=1)
    tk_npoints_per_edge.configure(width=60)
    # create a button to plot the 3D points
    irow += 1
    def plot3d_test_synthetic3dPoints():
        # convert string of 3 numbers to a tuple of 3 numbers
        # for example "1.0 2.0 3.0" to (1.0, 2.0, 3.0). The separator could be a space or a comma.
        center = list(map(float, tk_center.get().replace(',', ' ').split()))
        edges = list(map(float, tk_edges.get().replace(',', ' ').split()))
        npoints_per_edge = list(map(int, tk_npoints_per_edge.get().replace(',', ' ').split()))
        pos3Ds = create_synthetic_3d_points_cube2(center, edges, npoints_per_edge)
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(pos3Ds[:, 0], pos3Ds[:, 1], pos3Ds[:, 2])
        plt.show()
    tk_button = tk.Button(root, text='Plot 3D points', command=plot3d_test_synthetic3dPoints)
    tk_button.grid(row=irow, column=0)
    root.mainloop()


    