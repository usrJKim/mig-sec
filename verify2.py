import os
import torch

# Set the environment variable before importing any CUDA libraries
os.environ["CUDA_VISIBLE_DEVICES"] = "MIG-21ee95a0-0fcc-5bcb-9dfa-37910dc301f0"

# Now import torch or any other CUDA-dependent library
import torch

# Verify the current device
print(f"Current CUDA device: {torch.cuda.current_device()}")
