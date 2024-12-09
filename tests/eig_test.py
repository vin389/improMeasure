import numpy as np
import scipy

k = np.array([2,-1,-1,1], dtype=float).reshape(2,2)
m = np.array([1,0,0,1], dtype=float).reshape(2,2)

w, v = scipy.linalg.eig(k, m)

print('Stiffness matrix:', k)
print('Mass matrix:', m)
print('Eigenvalues:', w)
print('Eigenvectors:', v)

w1 = np.real(w[0])
w2 = np.real(w[1])
v1 = v[:,0]
v2 = v[:,1]


print('The end')