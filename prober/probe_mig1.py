#!/usr/bin/env python3
import os
import time
import pynvml
from pynvml import (
    nvmlInit, nvmlShutdown,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetSupportedMemoryClocks,
    nvmlDeviceGetSupportedGraphicsClocks,
    nvmlDeviceGetClock,
    nvmlDeviceGetPowerUsage,
    NVML_CLOCK_SM,
    NVML_CLOCK_ID_CURRENT
)

def main():
    # Probe interval (seconds, can be float), override via PROBE_INTERVAL
    interval = float(os.getenv("PROBE_INTERVAL", "0.1"))

    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)

    # (Optional) show supported clocks
    mem_clock = nvmlDeviceGetSupportedMemoryClocks(handle)[0]
    sm_clocks = nvmlDeviceGetSupportedGraphicsClocks(handle, mem_clock)
    #print(f"[Probe] MEM clocks: {nvmlDeviceGetSupportedMemoryClocks(handle)}")
    #print(f"[Probe] SM clocks @ MEM={mem_clock}: {sm_clocks}")

    try:
        while True:
            #current_sm = nvmlDeviceGetClock(handle, NVML_CLOCK_SM, NVML_CLOCK_ID_CURRENT)
            power_mw = nvmlDeviceGetPowerUsage(handle)
            #print(f"[Probe] Current SM clock: {current_sm} MHz")
            print(f"[Probe] Current power usage: {power_mw} mW")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        nvmlShutdown()

if __name__ == "__main__":
    #os.environ["CUDA_VISIBLE_DEVICES"] = "MIG-a6ef32aa-a74a-5c25-a183-48492dd3cd49"
    main()
