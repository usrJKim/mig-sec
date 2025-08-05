import pandas as pd
import matplotlib.pyplot as plt

file1 = "./fixed_power_edit.csv"
file2 = "./test_out_edit.csv"

df = pd.read_csv(file1)

df['wall_time_str'] = pd.to_datetime(df['wall_time_str'])

fig, ax1 = plt.subplots(figsize=(12,6))

color1 = "tab:blue"
ax1.set_xlabel("Wall time")
ax1.set_ylabel("Power (W)", color=color1)
ax1.plot(df["wall_time_str"], df["power_w"], color=color1, label="Power (W) of prober")
ax1.tick_params(axis="y", labelcolor=color1)

df = pd.read_csv(file2)

df['wall_time_str'] = pd.to_datetime(df['wall_time_str'])

ax2 = ax1.twinx()
color2 = "tab:red"
ax2.set_ylabel("Symbol", color=color2)
ax2.plot(df["wall_time_str"], df["symbol"], color=color2, alpha=0.5, label="Symbol")
ax2.tick_params(axis="y", labelcolor=color2)

plt.title("Power and Symbol over Time")
fig.tight_layout()
plt.grid(True)
plt.savefig("output.png")
