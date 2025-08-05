import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

file = "./fixed_power.csv"

df = pd.read_csv(file)

thres = 10
change_position=[]
last_max = df["power_w"].iloc[0]    

for i in range(1, len(df)):
    current = df["power_w"].iloc[i]
    if last_max - current >= thres:
        change_position.append(i)
        last_max = current
    elif current > last_max:
        last_max = current

block_ranges = []
start = 0
for point in change_position:
    block_ranges.append((start,point))
    start = point
block_ranges.append((start, len(df)-1))

#find gap
block_gap_values = []
for i, (start, end) in enumerate(block_ranges):
    block_gap = df["power_w"].iloc[end-1]-df["power_w"].iloc[end]
    block_gap_values.append((i, block_gap))

block_df = pd.DataFrame(block_gap_values, columns = ["block", "gap_power"])

# === 선형 회귀 ===
X = block_df["block"].values.reshape(-1, 1)
y = block_df["gap_power"].values
reg = LinearRegression().fit(X, y)
slope = reg.coef_[0]
intercept = reg.intercept_
y_pred = reg.predict(X)

plt.figure(figsize=(12,6))
plt.plot(block_df["block"], block_df["gap_power"])
plt.plot(block_df["block"], y_pred, linestyle="--", color="red", label=f"Linear Fit (slope={slope:.2f})")
plt.title("Gap Power per Block")
plt.xlabel("Block number")
plt.ylabel("Gap Power (w)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("output.png")
