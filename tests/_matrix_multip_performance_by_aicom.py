import numpy as np
from multiprocessing import Pool
import time

# Define the matrix dimensions
N = 1024
M = 1024
K = 1024

# Define the function for matrix multiplication
def matrix_multiply(args):
    a, b, i, j = args
    c_value = 0.0
    for k in range(M):
        c_value += a[i, k] * b[k, j]
    return c_value

# Generate random matrices to multiply
a = np.random.rand(N, M).astype(np.float64)
b = np.random.rand(M, K).astype(np.float64)
c = np.zeros((N, K)).astype(np.float64)

# Define the number of processes to use
num_processes = 4

# Divide the matrix into blocks for each process to handle
blocks = []
block_size = N // num_processes
for i in range(num_processes):
    for j in range(num_processes):
        block = (i * block_size, j * block_size)
        blocks.append(block)

# Warm-up the CPU by running the function once
args = [(a, b, blocks[0][0] + i, blocks[0][1] + j) for i in range(block_size) for j in range(block_size)]
results = list(map(matrix_multiply, args))

# Run the function multiple times and measure the performance
num_runs = 10
start_time = time.time()
for i in range(num_runs):
    pool = Pool(processes=num_processes)
    args = [(a, b, block[0] + i, block[1] + j) for block in blocks for i in range(block_size) for j in range(block_size)]
    results = pool.map(matrix_multiply, args)
    pool.close()
    pool.join()
end_time = time.time()

# Print the performance metrics
total_time = end_time - start_time
avg_time = total_time / num_runs
flops = (2.0 * N * M * K) / avg_time
print(f"Average time per run: {avg_time:.6f} s")
print(f"Total time for {num_runs} runs: {total_time:.6f} s")
print(f"GFLOPS: {flops/1e9:.6f}")
