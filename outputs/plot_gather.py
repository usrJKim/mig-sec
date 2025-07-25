import pandas as pd
import matplotlib.pyplot as plt
import glob

csv_files = glob.glob("./editted/*.csv")

plt.figure(figsize=(12,6))

for file in csv_files:
    df = pd.read_csv(file)
    label = file.replace(".csv", "")
    label = label.replace("./editted/", "")
    plt.plot(df['time_ms'], df['power_w'], label=label)

plt.xlabel("Time (msec)")
plt.ylabel('Power (W)')
plt.title('Power Consumption over Time(MIG 1g.10GB)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("summary_plot.png")
