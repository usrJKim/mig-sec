#!/usr/bin/env python3
import argparse
import time

import matplotlib.pyplot as plt
from pynvml import (
    nvmlInit,
    nvmlShutdown,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetPowerUsage
)

def main():
    parser = argparse.ArgumentParser(
        description="Probe GPU power usage and save a time-series plot."
    )
    parser.add_argument(
        "-i", "--interval",
        type=float,
        default=1.0,  # Set to 1ms or lower for higher resolution
        help="Probe interval in milliseconds"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="power_usage.png",
        help="Filename for the saved plot"
    )
    parser.add_argument(
        "-d", "--device",
        type=int,
        default=0,
        help="CUDA device index to monitor"
    )
    args = parser.parse_args()

    # Convert the interval from milliseconds to seconds
    interval_in_seconds = args.interval / 1000.0

    # Initialize NVML and get handle
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(args.device)

    timestamps = []
    powers = []

    start_t = time.perf_counter()
    next_t  = start_t

    try:
        while True:
            now = time.perf_counter()
            # Read power in milliwatts, convert to watts
            p_mw = nvmlDeviceGetPowerUsage(handle)
            timestamps.append(now - start_t)
            powers.append(p_mw / 1000.0)

            # compute next wakeup time and sleep
            next_t += interval_in_seconds
            to_sleep = next_t - time.perf_counter()
            if to_sleep > 0:
                time.sleep(to_sleep)
    except KeyboardInterrupt:
        # User stopped sampling
        pass
    finally:
        nvmlShutdown()

    # Plot and save
    plt.figure()
    plt.plot(timestamps, powers, linewidth=1)
    plt.xlabel("Time (s)")
    plt.ylabel("Power (W)")
    plt.title(f"GPU {args.device} Power Usage")
    plt.tight_layout()

    # Place timestamps on points where power > 70W, only on the first occurrence
    previous_above_threshold = False
    for i in range(len(powers)):
        if powers[i] > 70.0 and not previous_above_threshold:
            plt.text(timestamps[i], powers[i], f'{timestamps[i]:.3f}s', fontsize=4, ha='right')
            previous_above_threshold = True
        elif powers[i] <= 68.0:
            previous_above_threshold = False

    plt.savefig(args.output, dpi=500)
    print(f"Saved plot to {args.output!r}")

if __name__ == "__main__":
    main()
