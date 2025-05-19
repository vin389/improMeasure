import numpy as np

def deformation_2d(P, Q):
    """
    Compute the 2D deformation between two sets of points P and Q.
    P and Q should be 2D arrays of shape (4, 2) or (2, 4) float arrays.
    P is the original coordinates of 4 corners of the quadrilateral, and
    Q is the deformed coordinates of the same quadrilateral.
    The function returns the translation (ux, uy), rotation angle (theta, in degrees),
    and strains (ex, ey, gamma) in the x and y directions.
    The strains are engineering strains, defined as:
        ex = (Lx - L0) / L0
        ey = (Ly - L0) / L0
    where Lx and Ly are the lengths of the deformed quadrilateral, and L0 is the length of the original quadrilateral.
    gamma is the engineering shear strain.
    
    """
    P = np.asarray(P)
    Q = np.asarray(Q)
    if P.shape == (2, 4):
        P = P.T
    if Q.shape == (2, 4):
        Q = Q.T
    
    # Step 1: compute centroids
    c_P = np.mean(P, axis=0)
    c_Q = np.mean(Q, axis=0)
    
    # Center coordinates
    P_c = P - c_P
    Q_c = Q - c_Q

    # Step 2: best-fit affine transform A
    A, _, _, _ = np.linalg.lstsq(P_c, Q_c, rcond=None)

    # Step 3: polar decomposition A = R D
    U, S, Vt = np.linalg.svd(A)
    R = U @ Vt
    D = Vt.T @ np.diag(S) @ Vt

    # Step 4: extract theta and strains
    theta = np.arctan2(R[1, 0], R[0, 0]) * 180 / np.pi
    ux, uy = c_Q - R @ c_P
    ex = D[0, 0] - 1
    ey = D[1, 1] - 1
    gamma = D[0, 1] + D[1, 0]  # engineering shear strain
    #
    return ux, uy, theta, ex, ey, gamma


def deformation_3d(P, Q):
    P = np.asarray(P)  # shape (3, 4)
    Q = np.asarray(Q)  # shape (3, 4)
    
    if P.shape != (3, 4) or Q.shape != (3, 4):
        raise ValueError("P and Q must be 3x4 matrices")

    # Step 1: compute centroid of P
    centroid = np.mean(P, axis=1, keepdims=True)  # shape (3,1)
    P_centered = P - centroid

    # Step 2: compute best-fit plane using PCA (SVD)
    U, S, Vt = np.linalg.svd(P_centered)
    plane_normal = U[:, 2]  # third column is normal vector

    # Step 3: define local 2D basis on the plane
    u = U[:, 0]  # x-axis in plane
    v = U[:, 1]  # y-axis in plane
    basis = np.stack([u, v], axis=1).T  # shape (2, 3)

    # Step 4: project P and Q onto the 2D plane
    P2D = basis @ (P - centroid)  # shape (2,4)
    Q2D = basis @ (Q - centroid)  # shape (2,4)

    # Step 5: estimate strain from 2D deformation
    ux, uy, theta, ex, ey, gamma = deformation_2d(P2D, Q2D)
    #
    return ux, uy, theta, ex, ey, gamma

def unit_test_deformation_2d_rotation():
    P = np.array([[-1,-1], [+1,-1], [+1,+1], [-1,+1]])
    theta = 90. * np.pi / 180 # 5 degrees
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    Q = P @ R.T    
    ux, uy, theta, ex, ey, gamma = deformation_2d(P, Q)
    print("P:", P)
    print("Q:", Q)
    print(f"ux = {ux:.4f}, uy = {uy:.4f}")
    print(f"theta = {theta:.4f} degrees")
    print(f"strain_x = {ex:.4f}, strain_y = {ey:.4f}, shear_strain = {gamma:.4f}")

def unit_test_deformation_2d_simple_small_shear():
    P = np.array([[-1,-1], [+1,-1], [+1,+1], [-1,+1]])
    Q = P + np.array([[0,0],[0,0.001],[0,0.001],[0,0]])   
    ux, uy, theta, ex, ey, gamma = deformation_2d(P, Q)
    print("P:", P)
    print("Q:", Q)
    print(f"ux = {ux:.4f}, uy = {uy:.4f}")
    print(f"theta = {theta:.4f} degrees")
    print(f"strain_x = {ex:.4f}, strain_y = {ey:.4f}, shear_strain = {gamma:.4f}")


def unit_test_deformation_3d_simple_saddle_zero():
    P = np.array([[-1,-1,0], [+1,-1,0], [+1,+1,0], [-1,+1,0]], dtype=float).T
    d = 0.01
    Q = P + np.array([[0,0,d],[0,0,-d],[0,0,d],[0,0,-d]]).T
    ux, uy, theta, ex, ey, gamma = deformation_3d(P, Q)
    print("P:", P)
    print("Q:", Q)
    print(f"ux = {ux:.4f}, uy = {uy:.4f}")
    print(f"theta = {theta:.4f} degrees")
    print(f"strain_x = {ex:.4f}, strain_y = {ey:.4f}, shear_strain = {gamma:.4f}")



if __name__ == "__main__":
#    unit_test_deformation_2d_rotation()
#    unit_test_deformation_2d_simple_small_shear()
    unit_test_deformation_3d_simple_saddle_zero()