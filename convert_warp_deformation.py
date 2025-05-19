import numpy as np

def warp_to_deformation(warp, x0, y0, xc, yc, w, h):
    """
    Convert a warp to a deformation. Note: the deformation depends on the size and center 
    of the region of interest.
    
    Parameters:
        warp (ndarray): a 3x3 matrix representing the warp. If it is 2x3, it will be 
           converted to a 3x3 matrix by adding a row of [0, 0, 1].
        x0 (float): x-coordinate of the top-left corner of the region of interest.
        y0 (float): y-coordinate of the top-left corner of the region of interest.
        xc (float): x-coordinate of the center of the region of interest.
        yc (float): y-coordinate of the center of the region of interest.
        w (int): width of the region of interest.
        h (int): height of the region of interest.

    Returns:
        deformation (ndarray): 2D array representing the deformation.
        The deformation is a 8x1 vector, which is composed of:
        [ux, uy, exx, eyy, gamma_xy, bend_x, bend_y]. 

    Example 1: 
        warp = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        x0 = 0
        y0 = 0
        xc = 0
        yc = 0
        w = 2
        h = 2
        deformation = warp_to_deformation(warp, x0, y0, xc, yc, w, h)
        print(deformation)
        # Output: [[0. 0. 0. 0. 0. 0. 0. 0.]]

    Example 2: 
        warp = np.array([[1, 0, 0], [0, 1, 0]])
        x0 = 0
        y0 = 0
        xc = 0
        yc = 0
        w = 2
        h = 2
        deformation = warp_to_deformation(warp, x0, y0, xc, yc, w, h)
        print(deformation)
        # Output: [[0. 0. 0. 0. 0. 0. 0. 0.]]
    """
    # if warp is 2x3, convert it to 3x3 by adding a row of [0, 0, 1]
    if warp.shape == (2, 3):
        warp = np.vstack((warp, [0, 0, 1]))
    elif warp.shape == (3, 3):
        pass
    else:
        # print error message (including the function name) and raise exception
        print("# Error in warp_to_deformation: warp must be a 2x3 or 3x3 matrix")
        print("# warp and its shape: ", warp, warp.shape)
        raise ValueError("# warp must be a 2x3 or 3x3 matrix")
    
    # generate a 3x4 matrix. Each column is the coordinates of the four corners of the region of interest.
    # The first two columns are the coordinates of the top-left and bottom-left corners,
    # and the last two columns are the coordinates of the bottom-right and top-right corners.
    # the top-left corner is (x0, y0), the bottom-left corner is (x0, y0 + h - 1),
    # the bottom-right corner is (x0 + w - 1, y0 + h - 1), and the top-right corner is (x0 + w - 1, y0).
    # The last row is [1, 1, 1, 1] to make it a 3x4 matrix.
    p_before = np.array([[x0, x0, x0 + w - 1, x0 + w - 1],
                  [y0, y0 + h - 1, y0 + h - 1, y0], 
                  [1, 1, 1, 1]], dtype=float)
    # calculate the coordinates of the four corners of the region of interest after the warp
    p_after = warp @ p_before
    # divide the first two rows by the last row to get the coordinates of the four corners of the region of interest
    # and make the last row 1
    p_after = p_after[:2, :] / p_after[2, :]
    p_after = np.vstack((p_after, [1, 1, 1, 1]))
    # calculate the displacement by subtracting the coordinates of the four corners of the region of interest
    # and make it a 8x1 vector (ux1, uy1, ux2, uy2, ux3, uy3, ux4, uy4)
    disp = (p_after[:2,:] - p_before[:2,:]).T.reshape(8, 1)
    # calculate the deformation from the displacement
    d88 = mat_8dof_q4_deformation_to_displacement(x0, y0, xc, yc, w, h)
    d88_inv = np.linalg.inv(d88)
    deformation = d88_inv @ disp
    # return the deformation as a 8x1 vector
    return deformation.reshape(8, 1)
    

def deformation_to_warp(deformation, x0, y0, xc, yc, w, h):
    """
    Convert a deformation to a warp. Note: the deformation depends on the size and center 
    of the region of interest.
    
    Parameters:
        deformation (ndarray): a 8x1 array representing the deformation.
        The deformation is a 8x1 vector, which is composed of:
        [ux, uy, exx, eyy, gamma_xy, bend_x, bend_y]. 
        x0 (float): x-coordinate of the top-left corner of the region of interest.
        y0 (float): y-coordinate of the top-left corner of the region of interest.
        xc (float): x-coordinate of the center of the region of interest.
        yc (float): y-coordinate of the center of the region of interest.
        w (int): width of the region of interest.
        h (int): height of the region of interest.

    Returns:
        warp (ndarray): a 3x3 matrix representing the warp. If it is 2x3, it will be 
           converted to a 3x3 matrix by adding a row of [0, 0, 1].
    """
    # calculate the warp from the deformation
    # ux = deformation[0]
    # uy = deformation[1]
    # exx = deformation[2]
    # eyy = deformation[3]
    # gamma_xy = deformation[4]
    # bend_x = deformation[5]
    # bend_y = deformation[6]
    # calculate the warp from the deformation
    m = mat_8dof_q4_deformation_to_displacement(xc, yc, x0, y0, w, h)
    u = m @ (deformation.reshape(8, 1))
    # higher order terms of rotation (deformation[2])


    u_2x4 = (u.reshape(4,2)).T
    # calculate the coordinates before deformation (p_old_2x4)
    p_old_3x4 = np.array([[x0, x0, x0 + w - 1, x0 + w - 1],
                    [y0, y0 + h - 1, y0 + h - 1, y0], 
                    [1,1,1,1]], dtype=float)
    p_new_3x4 = p_old_3x4 + \
                np.array([[u_2x4[0,0], u_2x4[0,1], u_2x4[0,2], u_2x4[0,3]],
                    [u_2x4[1,0], u_2x4[1,1], u_2x4[1,2], u_2x4[1,3]], 
                    [0,0,0,0]], dtype=float)     
    # make it 
    # add a row of [1,1,1,1] to make it a homogeneous form (3x4 matrix)
    u_3x4 = np.vstack((u_2x4, [1, 1, 1, 1])) 
    # as p_new_3x4 = warp @ p_old_3x4, we can get the warp by
    # pn=w * po 
    # pn * po.T = w * po * po.T
    # pn * po.T * (po * po.T)^-1 = w * po * po.T * (po * po.T)^-1 
    # w = pn * po.T * (po * po.T)^-1 
    warp = p_new_3x4 @ p_old_3x4.T @ np.linalg.inv(p_old_3x4 @ p_old_3x4.T)    
    pass
    return warp


def mat_8dof_q4_deformation_to_displacement(xc, yc, x0, y0, w, h):
    m = np.zeros((8, 8), dtype=float)
    # ux
    m[0,:] = [1, 0, 1, 0, 1, 0, 1, 0]
    # uy
    m[1,:] = [0, 1, 0, 1, 0, 1, 0, 1]
    # rotation
    xr1 = xr2 = x0 - xc
    xr3 = xr4 = xr1 + w - 1
    yr1 = yr4 = y0 - yc
    yr2 = yr3 = yr1 + h - 1
    m[2,:] = [-yr1, xr1, -yr2, xr2, -yr3, xr3, -yr4, xr4]
    # strain xx 
    m[3,:] = [xr1, 0, xr2, 0, xr3, 0, xr4, 0]
    # strain yy 
    m[4,:] = [0, yr1, 0, yr2, 0, yr3, 0, yr4]
    # gamma xy (shear strain) (to be double checked)
    m[5,:] = [yr1, xr1, yr2, xr2, yr3, xr3, yr4, xr4]
    # bending mode (x)
    m[6,:] = [-xr1, 0, xr2, 0, xr3, 0, -xr4, 0]
    # bending mode (y)
    m[7,:] = [0, yr1, 0, yr2, 0, -yr3, 0, -yr4]
    m = m.T
    return m

def unit_test_mat88():
    # Example usage
    xc, yc = 0, 0
    x0, y0 = -0.5, -0.5
    w, h = 2, 2
    m = mat_8dof_q4_deformation_to_displacement(xc, yc, x0, y0, w, h)
    print("Matrix m:", m)
    m_inv = np.linalg.inv(m)
    print("Inverse of m:", m_inv)
    u = np.array([0, 0, 0., 0., 0., 0.001, 0, 0], dtype=float).reshape(8, 1)
    p1 = m @ u
    print(p1)

def unit_test_warp_deformation_zero():
    warp = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    x0 = 0
    y0 = 0
    xc = 0
    yc = 0
    w = 2
    h = 2
    deformation = warp_to_deformation(warp, x0, y0, xc, yc, w, h)
    print(deformation)

def unit_test_warp_deformation_rotate():
#    x0, y0, xc, yc, w, h = 200, 100, 205, 105, 10, 10
    x0, y0, xc, yc, w, h = -100, -100, 0, 0, 201, 201
    # rotation angle in radians
    th = 5. * np.pi / 180 # 5 degrees
    deform = np.array([[0, 0, th, 0, 0, 0, 0, 0]], dtype=float).T
    # convert deformation to warp
    warp = deformation_to_warp(deform, x0, y0, xc, yc, w, h)
    print("Warp matrix:", warp)
    # coordinate matrix P is a 3x4 matrix.
    p_old = np.array([[x0, x0, x0 + w - 1, x0 + w - 1],
                  [y0, y0 + h - 1, y0 + h - 1, y0], 
                  [1, 1, 1, 1]], dtype=float)
    # calculate the coordinates of the four corners of the region of interest after the warp
    p_new = warp @ p_old
    # divide the first two rows by the last row to get the coordinates of the four corners of the region of interest
    p_new_xy = p_new[:2, :] / p_new[2, :]
    # plot the original and warped points as red and blue quadrilaterals, respectively.
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(p_old[0, :], p_old[1, :], 'ro-', label='Original')
    plt.plot(p_new_xy[0, :], p_new_xy[1, :], 'bo-', label='Warped')
    plt.xlim(-100, 500)
    plt.ylim(-100, 500) 
    plt.gca().set_aspect('equal', adjustable='box')
    # The y-axis is flipped to match the image coordinate system.
    plt.gca().invert_yaxis()
    plt.title('Warping')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.grid()
    plt.show()
    pass





if __name__ == "__main__":
    # unit_test_mat88()
    # unit_test_warp_deformation_zero()
    unit_test_warp_deformation_rotate()