import numpy as np

# This function creates random 3D points within a given sphere. The sphere is defined 
# by a given point and a radius. 
# Input:
#   center: the center of the sphere, represented by a tuple (xc, yc, zc), 
#           a list [xc, yc, zc], or a numpy vector np.array([xc, yc, zc]).
#           The default value is (0.0, 0.0, 0.0).
#   radius: the radius of the sphere. The default is 1.0
#   nPoints: the number of points to generate. The default is 10.
# Output:
#   objPoints: a nPoints x 3 numpy array, where nPoints is the number of points.
#              Each row of pos3Ds is a 3D point.
# Example:
#   center = (0.0, 0.0, 0.0)
#   radius = 1.0
#   nPoints = 10
#   objPoints = create_synthetic_3d_points(center, radius, nPoints)
#   print(objPoints)
def create_synthetic_3d_points(center=(0.0, 0.0, 0.0), radius=1.0, nPoints=10):
    # generate random 3D points in a unit sphere
    pos3Ds = np.random.randn(nPoints, 3)
    # normalize the points
    pos3Ds = pos3Ds / np.linalg.norm(pos3Ds, axis=1)[:, None]
    # multiply by a random scale
    pos3Ds = pos3Ds * np.random.rand(nPoints)[:, None]
    # multiply by the radius of the sphere
    pos3Ds = pos3Ds * radius
    # add the center of the sphere
    pos3Ds = pos3Ds + np.array(center)
    return pos3Ds

# This function plots the 3D points in a 3D plot using matplotlib for user to verify the 3D points.
# Input:
#   pos3Ds: a nPoints x 3 numpy array, where nPoints is the number of points.
#           Each row of pos3Ds is a 3D point.
#   center: the center of the sphere for visual verification, represented by a tuple (xc, yc, zc). The default is None.
#   radius: the radius of the sphere for visual verification. The default is 0.0
#   theAxes: the axes of the plot. The default is None. If None, a new figure will be created.
#            If it is not None, the plot will be added to the existing figure.
# Example:
#   center = (0.0, 0.0, 0.0)
#   radius = 1.0
#   nPoints = 10
#   pos3Ds = create_synthetic_3d_points(center, radius, nPoints)
#   plot_3dPoints_and_sphere(pos3Ds, center, radius)
def plot_3dPoints_and_sphere(pos3Ds, center=None, radius=0.0, theAxes=None):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    # create a new figure if theAxes is None
    if theAxes is None:
        fig = plt.figure()
        theAxes = fig.add_subplot(111, projection='3d')
    ax = theAxes
    # plot the 3D points
    ax.scatter(pos3Ds[:, 0], pos3Ds[:, 1], pos3Ds[:, 2])
    if center is not None and radius > 0:
        # draw a semi-transparent sphere in matplotlib 
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
        y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
        z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]
        ax.plot_surface(x, y, z, color='green', alpha=0.1)
    plt.show(block=False)
    return theAxes



# This function is a unit test of synthetic3dPoints()
# It creates a tkinter window with a text entry for sphere center
# and a text entry for sphere radius, and a text entry for the number of points.
# It also creates a button to generate the 3D points by calling synthetic3dPoints.
# When the button is clicked, it calls synthetic3dPoints() to generate 3D points
# Then it popups a window to display the 3D points by using matplotlib. 
def test_synthetic3dPoints():
    import tkinter as tk
    from tkinter import simpledialog
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    def plot3d_test_synthetic3dPoints():
        center = centerEntry.get()
        radius = radiusEntry.get()
        npoints = npEntry.get()
        try:
            center = eval(center)
            radius = float(radius)
            npoints = int(npoints)
            pos3Ds = create_synthetic_3d_points(center, radius, npoints)
            plot_3dPoints_and_sphere(pos3Ds, center, radius)
        except Exception as e:
            print("Error: ", e)

    root = tk.Tk()
    # set the title of the tk window "Synthetic 3D Points "
    root.title("Synthetic 3D Points" + " "*20 + "Created Github copilot under supervision of Vince Yang")
    # make the width of this window 500 pixels and the height 200 pixels
    root.geometry("500x200")
    centerLabel = tk.Label(root, text="Center of the sphere (x, y, z):")
    centerLabel.pack()
    centerEntry = tk.Entry(root)
    centerEntry.pack()
    centerEntry.insert(0, "(0.0, 0.0, 0.0)")
    radiusLabel = tk.Label(root, text="Radius of the sphere:")
    radiusLabel.pack()
    radiusEntry = tk.Entry(root)
    radiusEntry.pack()
    radiusEntry.insert(0, "1.0")
    npLabel = tk.Label(root, text="Number of points:")
    npLabel.pack()
    npEntry = tk.Entry(root)
    npEntry.pack()
    npEntry.insert(0, "10")
    # create a text entry for xlim of the plot
    label = tk.Label(root, text="xlim of the plot:")
    label.pack()
    xlimEntry = tk.Entry(root)
    xlimEntry.pack()
    xlimEntry.insert(0, "(-10, 10)")
    # create a text entry for ylim of the plot
    label = tk.Label(root, text="ylim of the plot:")
    label.pack()
    ylimEntry = tk.Entry(root)
    ylimEntry.pack()
    ylimEntry.insert(0, "(-10, 10)")
    # create a text entry for zlim of the plot
    label = tk.Label(root, text="zlim of the plot:")
    label.pack()
    zlimEntry = tk.Entry(root)
    zlimEntry.pack()
    zlimEntry.insert(0, "(-10, 10)")

    button = tk.Button(root, text="Plot 3D", command=plot3d_test_synthetic3dPoints)
    button.pack()
    root.mainloop()

# main
if __name__ == '__main__':
    test_synthetic3dPoints()
    pass