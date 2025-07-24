#!/usr/bin/env python3
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Plot GPU power and annotate drops from >80 W to <70 W")
    parser.add_argument(
        "-i", "--input",
        type=str,
        default="power_data.csv",
        help="CSV file with columns: time_ms,power_w"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="power_drop_plot.png",
        help="Filename for saved plot"
    )
    args = parser.parse_args()

    # Read CSV
    df = pd.read_csv(args.input)
    if "time_ms" not in df or "power_w" not in df:
        raise ValueError("Input CSV must have columns time_ms,power_w")

    # Convert ms to seconds for plotting
    df["time_s"] = df["time_ms"] / 1000.0

    # Find drop events: previous >80 and current <70
    cond = (df["power_w"].shift(1) > 80) & (df["power_w"] < 70)
    drops = df[cond]

    # Print timestamps in ms
    print("Detected drops from >80 W to <70 W at:")
    for tm in drops["time_ms"]:
        print(f"  {tm} ms")

    # Plot
    plt.figure(figsize=(8,4))
    plt.plot(df["time_s"], df["power_w"], label="Power (W)", linewidth=1)

    # Annotate each drop
    for _, row in drops.iterrows():
        t_s = row["time_s"]
        pw  = row["power_w"]
        label = f"{int(row['time_ms'])} ms"
        plt.scatter([t_s], [pw], color="red", s=30)
        plt.annotate(label,
                     xy=(t_s, pw),
                     xytext=(5, -15),
                     textcoords="offset points",
                     fontsize=8,
                     color="red",
                     arrowprops=dict(arrowstyle="->", color="red"))

    plt.xlabel("Time (s)")
    plt.ylabel("Power (W)")
    plt.title("GPU Power Over Time\n(red markers show drops >80→<70 W)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output, dpi=300)
    print(f"Saved annotated plot to '{args.output}'")

if __name__ == "__main__":
    main()
