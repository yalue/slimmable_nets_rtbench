# This script is a very simply benchmark repeatedly adding two tensors. Just
# run it and see the times.

import torch
import time

stream = torch.cuda.Stream(device="cuda:0")

a = torch.rand((1000, 1000, 100), device="cuda:0")
b = torch.rand((1000, 1000, 100), device="cuda:0")
start_time = time.perf_counter()
with torch.cuda.stream(stream):
    c = a + b
    stream.synchronize()
end_time = time.perf_counter()
print("Warmup done in %fs." % (end_time - start_time,))

total_time = 0.0
min_time = 2.0e9
max_time = -2.0e9
tmp = 0.0
sample_count = 100

for i in range(sample_count):
    start_time = time.perf_counter()
    with torch.cuda.stream(stream):
        c = a + b
    stream.synchronize()
    end_time = time.perf_counter()
    tmp = end_time - start_time
    if tmp > max_time:
        max_time = tmp
    if tmp < min_time:
        min_time = tmp
    total_time += tmp

print("Average time to add 1000x1000x100 tensor: %f seconds." % (total_time / float(sample_count)))
print("Min: %f seconds, max: %f seconds." % (min_time, max_time))

