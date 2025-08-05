import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

file = "./fixed_power.csv"

df = pd.read_csv(file)

gaps = []
for i in range(len(df)-1):
    gap = df["power_w"].iloc[i+1] - df["power_w"].iloc[i] 
    if i > 700000:
        if gap > 0:
            gaps.append((i, gap))

gap_df = pd.DataFrame(gaps, columns = ["block", "step"])

# === 선형 회귀 ===
X = gap_df["block"].values.reshape(-1, 1)
y = gap_df["step"].values
reg = LinearRegression().fit(X, y)
slope = reg.coef_[0]
intercept = reg.intercept_
y_pred = reg.predict(X)

plt.figure(figsize=(12,6))
plt.plot(gap_df["block"], gap_df["step"])
plt.plot(gap_df["block"], y_pred, linestyle="--", color="red", label=f"Linear Fit (slope={slope:.2f}, intrcpt={intercept:.2f})")
plt.title("Power step from i to i+1 ms(W)")
plt.xlabel("Time (ms)")
plt.ylabel("Power step (w)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("output.png")
