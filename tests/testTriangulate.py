from triangulatePoints2 import triangulatePoints2
import numpy as np 

#
rvec1 = np.array([0,0,0])
tvec1 = np.array([0,0,0])
cmat1 = np.array(
    [[1.98888802e+03, 0.00000000e+00, 2.02921156e+03],
    [0.00000000e+00, 1.98210177e+03, 1.50853102e+03],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], 
    dtype=np.float64).reshape((3,3))
dvec1 = np.array([-0.00659889,  0.,  0.,  0.,  0.], dtype=np.float64)
#
cmat2 = np.array(
    [[1.92745626e+03, 0.00000000e+00, 2.02121427e+03],
       [0.00000000e+00, 1.92639892e+03, 1.51642305e+03],
       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], 
       dtype=np.float64).reshape((3,3))
dvec2 = np.array([-0.00291725,  0.,  0.,  0.,  0.], dtype=np.float64)
rvec2 = np.array([-0.02618424, -0.01501711,  0.00072262], dtype=np.float64)
tvec2 = np.array([-40.46213113,  72.02069942, -31.14572627], dtype=np.float64)
#
imgPoints1 = np.array([[2312.82994924, 1931.22357143],
             [2318.34864865, 1773.27428571],
             [2319.88673139, 1616.51428571]], dtype=np.float64)
imgPoints2 = np.array([[2049.38190955, 2411.83214286],
             [2051.91377091, 2245.1],
             [2050.73699422, 2078.48214286]], dtype=np.float64)
