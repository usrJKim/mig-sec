#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.naive_bayes     import GaussianNB
from sklearn.ensemble         import RandomForestClassifier
from sklearn.neighbors        import KNeighborsClassifier
from sklearn.svm              import SVC
from sklearn.linear_model     import LogisticRegression

def load_sequence(seq_csv, n_calib):
    """Read at least n_calib labels from sequence.csv, flattened."""
    with open(seq_csv, 'r') as f:
        text = f.read()
    tokens = text.replace('\n', ',').split(',')
    seq = [int(tok) for tok in tokens if tok.strip()]
    if len(seq) < n_calib:
        raise ValueError(f"Need {n_calib} labels, got {len(seq)}")
    return np.array(seq[:n_calib], dtype=int)

def load_power_df(power_csv):
    """Load millisecond-resolution power trace into DataFrame indexed by time_ms."""
    df = pd.read_csv(power_csv)
    if "time_ms" not in df or "power_w" not in df:
        raise ValueError("power CSV must have columns time_ms,power_w")
    return df.set_index("time_ms", drop=False)

def first_non_outlier(window_vals):
    """
    Identify 1.5×IQR outliers and return first in-range value, else median.
    """
    q1, q3 = np.percentile(window_vals, [25,75])
    iqr    = q3 - q1
    low, high = q1 - 1.5*iqr, q3 + 1.5*iqr
    for v in window_vals:
        if low <= v <= high:
            return v
    return np.median(window_vals)

def sample_windows(df, interval_ms, slack_ms, n_windows, t_start=None):
    """
    For each window i:
      - window covers [t_start + i*interval_ms, + slack_ms]
      - pick first non-outlier or median fallback
    Returns (picks, centers)
    """
    if t_start is None:
        t_start = int(df["time_ms"].min())
    picks, centers = [], []
    for i in range(n_windows):
        w0 = t_start + i*interval_ms
        w1 = w0 + slack_ms
        window = df.loc[(df["time_ms"]>=w0)&(df["time_ms"]<=w1), "power_w"].values
        if window.size == 0:
            idx  = df.index.get_indexer([w0], method="nearest")[0]
            pick = df.iloc[idx]["power_w"]
        else:
            pick = first_non_outlier(window)
        picks.append(pick)
        centers.append(w0 + slack_ms//2)
    return np.array(picks), np.array(centers)

def get_classifier(name):
    """Factory for classifiers (None means threshold-Gaussian)."""
    if name == "threshold":
        return None
    if name == "gnb":
        return GaussianNB()
    if name == "rf":
        return RandomForestClassifier(n_estimators=200, random_state=0)
    if name == "knn":
        return KNeighborsClassifier(n_neighbors=5)
    if name == "svc":
        return SVC(kernel="rbf", probability=True, random_state=0)
    if name == "logreg":
        return LogisticRegression(max_iter=500, random_state=0)
    raise ValueError(f"Unknown model '{name}'")

def main():
    p = argparse.ArgumentParser(
        description="Outlier-robust calibration & decoding with multiple classifiers"
    )
    p.add_argument("-p","--power",   default="power_data.csv",
                   help="CSV with columns time_ms,power_w")
    p.add_argument("-s","--sequence",default="./input_files/sequence.csv",
                   help="CSV of calibration preamble symbols")
    p.add_argument("-i","--interval",type=int, default=100,
                   help="Nominal interval between windows (ms)")
    p.add_argument("-l","--slack",   type=int, default=10,
                   help="Slack in each window (ms)")
    p.add_argument("-n","--nc",      type=int, default=150,
                   help="Number of calibration windows")
    p.add_argument("-m","--model",
                   choices=["threshold","gnb","rf","knn","svc","logreg"],
                   default="gnb", help="Classifier to use")
    p.add_argument("-o","--output",  default="decoded_following.csv",
                   help="Decoded output CSV")
    args = p.parse_args()

    # 1) Load data and calibration preamble
    df = load_power_df(args.power)
    cal_picks, cal_centers = sample_windows(
        df, args.interval, args.slack, args.nc
    )
    seq = load_sequence(args.sequence, args.nc)

    # 2) Visualize calibration preamble picks
    plt.figure(figsize=(12,4))
    plt.plot(cal_centers, cal_picks, '-o', markersize=4, label='Picked Power')
    for t, pw, lbl in zip(cal_centers, cal_picks, seq):
        plt.text(t, pw+0.2, str(lbl),
                 fontsize=8, ha='center', va='bottom', color='blue')
        if lbl == 0:
            plt.text(t, pw-0.5, f"{int(t)} ms",
                     fontsize=7, ha='center', va='top', color='red')
    plt.xlabel("Time (ms)")
    plt.ylabel("Power (W)")
    plt.title(f"Calibration Preamble (first {args.nc} windows\n{args.interval}±{args.slack} ms)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("calibration_plot.png", dpi=300)
    print("Saved calibration_plot.png")

    # 3) Train chosen classifier
    clf = get_classifier(args.model)
    if args.model == "threshold":
        stats = {}
        for lbl in np.unique(seq):
            vals = cal_picks[seq==lbl]
            stats[lbl] = (float(vals.mean()), float(vals.var()))
    else:
        clf.fit(cal_picks.reshape(-1,1), seq)

    # 4) Print calibration stats
    print(f"\nCalibration stats ({args.nc} windows):")
    for lbl in np.unique(seq):
        vals = cal_picks[seq==lbl]
        print(f"  Label {lbl:2d}: n={len(vals):3d}, "
              f"min={vals.min():.3f}, max={vals.max():.3f}, std={vals.std(ddof=0):.3f}")

    # 5) Decode remaining windows
    total_w = int(((df["time_ms"].max()-df["time_ms"].min())//args.interval)+1)
    all_picks, all_centers = sample_windows(
        df, args.interval, args.slack, total_w
    )
    decoded = []
    for pick, center in zip(all_picks[args.nc:], all_centers[args.nc:]):
        if args.model == "threshold":
            best_lbl, best_p = None, -1e18
            for lbl,(mu,var) in stats.items():
                p = -0.5*((pick-mu)**2)/var - 0.5*np.log(2*np.pi*var)
                if p > best_p:
                    best_p, best_lbl = p, lbl
            lbl_pred = best_lbl
        else:
            lbl_pred = int(clf.predict([[pick]])[0])
        decoded.append((int(center), float(pick), lbl_pred))

    # 6) Save decoded results
    out_df = pd.DataFrame(decoded, columns=["time_ms","power_w","decoded_label"])
    out_df.to_csv(args.output, index=False)
    print(f"\nDecoded {len(decoded)} windows → '{args.output}'")

if __name__ == "__main__":
    main()
