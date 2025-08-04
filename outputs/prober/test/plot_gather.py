import pandas as pd
import matplotlib.pyplot as plt
import glob
import argparse

plt.figure(figsize=(12,6))

df = pd.read_csv("old.csv")
plt.plot(df['time_ms'], df['power_w'], label="old",color="blue")

df = pd.read_csv("new.csv")
plt.plot(df['time_ms'], df['power_w'], label="new",color="orange")

plt.xlabel("Time (msec)")
plt.ylabel('Power (W)')
plt.title('Power Consumption over Time')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("./summary.png")
