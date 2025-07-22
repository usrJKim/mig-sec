#!/usr/bin/env python3
import os
import time
import pynvml
from pynvml import (
    nvmlInit, nvmlShutdown,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetSupportedMemoryClocks,
    nvmlDeviceGetSupportedGraphicsClocks,
    nvmlDeviceSetGpuLockedClocks,
)

def main():
    # Toggle period (seconds, can be float), override via TOGGLE_PERIOD
    period = float(os.getenv("TOGGLE_PERIOD", "0.1"))

    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)

    # Pick first supported mem clock, then get all supported SM clocks at that mem clock
    mem_clock = nvmlDeviceGetSupportedMemoryClocks(handle)[0]
    sm_clocks = nvmlDeviceGetSupportedGraphicsClocks(handle, mem_clock)
    #print(f"Supported SM clocks at MEM={mem_clock} MHz:  ", sm_clocks)
    # Alternate between lowest and highest supported SM clocks
    sm_freqs = [sm_clocks[0], sm_clocks[-1]]
    idx = 0

    try:
        while True:
            target_sm = sm_freqs[idx]
            try:
                nvmlDeviceSetGpuLockedClocks(handle, target_sm, target_sm)
                print(f"[Toggle] Locked MEM={mem_clock} MHz, SM={target_sm} MHz")
            except pynvml.NVMLError as e:
                print(f"[Toggle] Error setting locked clocks: {e}")
            idx = (idx + 1) % len(sm_freqs)
            time.sleep(period)
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        nvmlShutdown()

if __name__ == "__main__":
    #os.environ["CUDA_VISIBLE_DEVICES"] = "MIG-21ee95a0-0fcc-5bcb-9dfa-37910dc301f0"
    main()
