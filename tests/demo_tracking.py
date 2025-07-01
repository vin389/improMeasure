import numpy as np

f = np.zeros(30, dtype=int)
f[0] = 0; 
f[1] = 1;
for i in range(2,30):
    f[i] = f[i-1] + f[i-2]
print(f)
