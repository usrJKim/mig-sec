#!/usr/bin/env python3
"""
Reset locked GPU clocks on GPUs 0-3 using NVML.
Requires:
  pip install pynvml
Run with root (or CAP_SYS_ADMIN) privileges.
"""

import sys
import pynvml as nv

TARGET_GPUS = [0]

def main() -> None:
    try:
        nv.nvmlInit()
    except nv.NVMLError as e:
        sys.exit(f"Failed to initialise NVML: {e}")

    failed = []
    for idx in TARGET_GPUS:
        try:
            handle = nv.nvmlDeviceGetHandleByIndex(idx)
            nv.nvmlDeviceResetGpuLockedClocks(handle)      # SM/graphics clocks
            nv.nvmlDeviceResetApplicationsClocks(handle)
            name = nv.nvmlDeviceGetName(handle)#.decode()
            print(f"[GPU {idx}] {name}: locked clocks cleared")
        except nv.NVMLError as e:
            print(f"[GPU {idx}] NVML error: {e}")
            failed.append(idx)

    nv.nvmlShutdown()

    if failed:
        sys.exit(f"One or more resets failed: {failed}")
    print("All requested GPUs reset successfully.")

if __name__ == "__main__":
    main()