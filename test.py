from torch.profiler import profile, record_function
from network.layers import EdgeFeature
import torch
import time

model1 = EdgeFeature(k=16).to("cuda")
model2 = EdgeFeature(k=16, use_pointnet=True).to("cuda")

input_ = torch.randn(16, 3, 4000).to("cuda")

# Number of times to run each function
num_runs = 100

# Profile the EdgeFeature model
with profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
) as prof:
    start = time.time()
    for _ in range(num_runs):
        with record_function("model1"):
            tensor, idx = model1(input_)
    print(time.time() - start)
print(prof.key_averages())

# Profile the EdgeFeature2 model
with profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
) as prof:
    start = time.time()
    for _ in range(num_runs):
        with record_function("model2"):
            tensor, idx = model2(input_)
    print(time.time() - start)
print(prof.key_averages())
