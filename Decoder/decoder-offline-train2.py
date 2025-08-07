#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import joblib
import time

# Constants for windowing and bump correction
SAMPLE_OFFSET_MS = 35
LOOKAHEAD_MS = 10
DUAL_INTERVAL_OFFSET_MS = 20
DUAL_DIFF_THRESHOLD_PWR = 5.0


def load_power_df(path):
    df = pd.read_csv(path)
    required = {'time_ms','power_w','temp_C'}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must have columns {required}")
    return df.sort_values('time_ms').reset_index(drop=True)


def compute_symbol_starts(df, base, interval, total):
    """
    1. Compare power at s1 and s2 to pick the current symbol start.
    2. If s2 is chosen, look ahead at s2+(interval-20) and s2+interval:
       if their power difference exceeds threshold, shorten next interval by 20ms.
    """
    def sample_power(t):
        w = df[(df['time_ms'] >= t) & (df['time_ms'] < t + LOOKAHEAD_MS)]
        if len(w) > 0:
            return w.iloc[0]['power_w']
        idx = (np.abs(df['time_ms'] - t)).idxmin()
        return df.loc[idx, 'power_w']

    def adj(c):
        w = df[(df['time_ms'] >= c) & (df['time_ms'] < c + LOOKAHEAD_MS)]
        if len(w) >= 2:
            first = w.iloc[0]['power_w']
            diff = w[w['power_w'] != first]
            if not diff.empty:
                return int(diff.iloc[0]['time_ms'])
        return c

    starts = []
    prev = base
    current_interval = interval
    for i in range(total):
        nominal = prev + current_interval if i > 0 else base
        s1 = adj(nominal)
        s2 = adj(nominal + DUAL_INTERVAL_OFFSET_MS)

        p1 = sample_power(s1)
        p2 = sample_power(s2)
        if abs(p2 - p1) > DUAL_DIFF_THRESHOLD_PWR:
            cur = s2
            t80 = cur + interval - DUAL_INTERVAL_OFFSET_MS - 3
            t100 = cur + interval - 3
            if abs(sample_power(t100) - sample_power(t80)) > DUAL_DIFF_THRESHOLD_PWR:
                current_interval = interval - DUAL_INTERVAL_OFFSET_MS
            else:
                current_interval = interval
        else:
            cur = s1
            t80 = cur + interval - DUAL_INTERVAL_OFFSET_MS - 3
            t100 = cur + interval - 3
            if abs(sample_power(t100) - sample_power(t80)) > DUAL_DIFF_THRESHOLD_PWR:
                current_interval = interval - DUAL_INTERVAL_OFFSET_MS
            else:
                current_interval = interval

        starts.append(cur)
        prev = cur
    return starts


def window_averages(df, starts, window_ms):
    avgs, centers = [], []
    for st in starts:
        s, e = st + SAMPLE_OFFSET_MS, st + SAMPLE_OFFSET_MS + window_ms
        vals = df[(df['time_ms'] >= s) & (df['time_ms'] < e)]['power_w'].values
        if vals.size:
            q1, q3 = np.percentile(vals, [25, 75])
            iqr = q3 - q1
            lo, hi = q1 - 1.5*iqr, q3 + 1.5*iqr
            filt = vals[(vals >= lo) & (vals <= hi)]
            mu = filt.mean() if filt.size else vals.mean()
        else:
            mu = np.nan
        avgs.append(mu)
        centers.append(s + window_ms/2)
    return np.array(avgs), np.array(centers)


def load_seq(path):
    lines = open(path).read().splitlines()
    if len(lines) <= 1:
        raise ValueError("Label file must have header and data.")
    data = ','.join(lines[1:]).split(',')
    return [int(x) for x in data if x.strip()]


def main():
    p = argparse.ArgumentParser(description="Decode power trace, learn correlation, and visualize.")
    p.add_argument('-d', '--debug', action='store_true', help="Enable debug prints")
    p.add_argument('-p', '--power', default='power_data.csv', help="CSV: time_ms,power_w,temp_C")
    p.add_argument('-s', '--seq', default='./input_files/sequence.csv', help="Label file: skip header, one label per window")
    p.add_argument('-i', '--interval', type=int, default=100, help="Symbol interval in ms")
    p.add_argument('-w', '--window', type=int, default=30, help="Window size in ms for averaging")
    p.add_argument('--zoom', nargs=2, type=float, metavar=('START','END'),
                   help="If given, zoom into [START,END] ms on power plot")
    args = p.parse_args()

    df = load_power_df(args.power)
    base = int(df['time_ms'].min())
    init_total = int((df['time_ms'].max() - base) // args.interval) + 1

    labels_full = load_seq(args.seq)
    total = min(init_total, len(labels_full))
    if len(labels_full) < init_total:
        print(f"Warning: trimming windows {init_total}->{total} based on labels")

    starts = compute_symbol_starts(df, base, args.interval, total)
    avgs, centers = window_averages(df, starts, args.window)
    temps = np.interp(centers, df['time_ms'], df['temp_C'])

    prev_avgs  = np.concatenate([[avgs[0]],  avgs[:-1]])
    prev_temps = np.concatenate([[temps[0]], temps[:-1]])
    
    starts_arr = np.array(starts)
    X = np.column_stack((avgs, temps, prev_avgs, prev_temps))
    valid = ~np.isnan(X).any(axis=1)
    starts_arr, X = starts_arr[valid], X[valid]

    # Train random forest classifier
    y = np.array(labels_full[:total])[valid]
    '''
    X_mean, X_std = X.mean(axis=0), X.std(axis=0)
    z_scores = np.abs((X - X_mean) / X_std)
    z_mask = (z_scores < 3).all(axis=1)
    n_before = len(X)
    starts_arr_clean, X_clean, y_clean = starts_arr[z_mask], X[z_mask], y[z_mask]
    print(f"Using {n_before} -> {len(X_clean)} clean samples for training")

    if args.debug:
        dropped = np.where(~z_mask)[0]
        for i in dropped:
            print(
                f"  idx={i}, start_ms={int(starts_arr[i])}, "
                f"power_avg={X[i,0]:.3f}, temp_C={X[i,1]:.2f}, "
                f"prev_power={X[i,2]:.3f}, prev_temp={X[i,3]:.2f}"
            )
    '''
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    #clf = XGBClassifier(n_estimators=50, random_state=42)  # Using XGBoost for better performance

    #clf.fit(X_clean, y_clean)
    clf.fit(X, y)
    preds = clf.predict(X)
    print(f"Training accuracy: {accuracy_score(y, preds)*100:.2f}%")

    # save trained model
    os.makedirs('models', exist_ok=True)
    joblib.dump(clf, 'models/power_model2.pkl')
    print("Saved model -> models/power_model2.pkl")

    title = "Supervised RF"

    # Debug prints
    if args.debug:
        print("\nDebug info (start_ms, power_avg, temp, label):")
        for s, row in zip(starts_arr, np.column_stack((X, y))):
            p, t, l = row[0], row[1], int(row[-1])
            print(f"{int(s)} ms, {p:.3f} W, {t:.2f} °C, label={l}")

        print("\nStats per label (power mean±std, temp mean±std):")
        for lab in np.unique(y):
            m = (y == lab)
            pw, tp = X[m,0], X[m,1]
            print(f"label {lab}: power {pw.mean():.3f}±{pw.std():.3f} W, temp {tp.mean():.2f}±{tp.std():.2f} °C")

    # Scatter plot (sample up to 10 labels)
    unique = np.unique(y)
    np.random.seed(int(time.time()))
    sample = np.random.choice(unique, size=min(10, len(unique)), replace=False)
    counts = {lab: np.sum(y == lab) for lab in sample}
    sample_sorted = sorted(sample, key=lambda lab: counts[lab])
    fig, ax = plt.subplots(figsize=(6,6))
    for lab in sample_sorted:
        mask = (y == lab)
        ax.scatter(X[mask,1], X[mask,0], s=20, label=f"{lab}")
    ax.set_xlabel('Temp (°C)')
    ax.set_ylabel('Power (W)')
    ax.set_title(title)
    handles, labels = ax.get_legend_handles_labels()
    labels_int = [int(l) for l in labels]
    order = sorted(range(len(labels_int)), key=lambda i: labels_int[i])
    ax.legend([handles[i] for i in order], [labels[i] for i in order], fontsize=8,
              bbox_to_anchor=(1.05,1), loc='upper left')
    plt.tight_layout()
    plt.savefig('power_temp_rf.png', dpi=300)
    print("Saved plot -> power_temp_rf.png")

    if args.zoom:
        z0, z1 = args.zoom
        mask = (df['time_ms'] >= z0) & (df['time_ms'] <= z1)
        fig2, ax2 = plt.subplots(figsize=(8,4))
        ax2.plot(df.loc[mask, 'time_ms']/1000, df.loc[mask, 'power_w'], linewidth=1)
        for s in starts_arr:
            if z0 <= s <= z1:
                ax2.axvline(s/1000, linestyle='--', color='gray', alpha=0.7)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Power (W)')
        ax2.set_title(f"Zoomed power [{int(z0)}–{int(z1)}] ms")
        plt.tight_layout()
        plt.savefig('power_zoom.png', dpi=300)
        print("Saved zoom plot -> power_zoom.png")

if __name__ == '__main__':
    main()
