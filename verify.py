#!/usr/bin/env python3
import os
import time
import torch

def main():
    # 1) Bind to the first MIG instance by UUID
    os.environ["CUDA_VISIBLE_DEVICES"] = "MIG-21ee95a0-0fcc-5bcb-9dfa-37910dc301f0" #change this to your MIG UUID and verify the tensor is loaded to the correct MIG instance
    
    # 2) Pick a device
    device = torch.device("cuda:0")
    print(f"Running on device: {device}")

    # 3) Query total memory (in bytes) on this MIG instance
    total_mem = torch.cuda.get_device_properties(device).total_memory
    print(f"Total MIG memory: {total_mem/1e9:.2f} GB")

    # 4) Allocate ~80% of it as a single tensor
    alloc_bytes = int(total_mem * 0.2)
    num_floats = alloc_bytes // 4  # float32 = 4 bytes
    print(f"Allocating tensor with {num_floats:,} floats (~{alloc_bytes/1e9:.2f} GB)...")
    x = torch.empty((num_floats,), dtype=torch.float32, device=device)

    # Touch the tensor so allocation actually happens
    x.fill_(1.0)
    torch.cuda.synchronize()

    print("Allocation done. Holding for 5 minutes so you can check via nvidia-smi.")
    time.sleep(300)  # 5 minutes

    # 5) Cleanup
    del x
    torch.cuda.empty_cache()
    print("Freed tensor, exiting.")

if __name__ == "__main__":
    main()
