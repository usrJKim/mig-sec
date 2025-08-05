import pandas as pd
import matplotlib.pyplot as plt
import glob

def labelling(file):
    if 'gemv' in file:
        return 'GEMV', "#1f77b4"
    elif 'sharedMem' in file:
        return 'Shared memory',"#ff7f0e"
    elif 'sfu' in file:
        return 'SFU', "#2ca02c"
    elif 'fp' in file:
        return 'Floating point',"#d62728"

csv_files = glob.glob("./data/*.csv")
plt.figure(figsize=(12,6))

for file in csv_files:
    df = pd.read_csv(file)
    label,color = labelling(file)
    plt.plot(df['time_ms'], df['power_w'], label=label,color=color)

plt.xlabel("Time (msec)")
plt.ylabel('Power (W)')
plt.title('Power Consumption over Time')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("./summary_plot.png")
