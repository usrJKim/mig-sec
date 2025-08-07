#!/usr/bin/env python3
# take previous value as an another feature

import os
import argparse
import pandas as pd
import numpy as np
import joblib
import time

# Constants for windowing and bump correction (must match training code)
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
    parser = argparse.ArgumentParser(
        description="Load pretrained model and decode continuous power/temperature trace."
    )
    parser.add_argument('-p','--power', default='power_data.csv',
                        help="CSV with columns time_ms,power_w,temp_C")
    parser.add_argument('-s','--seq', default='./input_files/sequence.csv',
                        help="Ground-truth label file (skip header)")
    parser.add_argument('-m','--model', default='models/power_model2.pkl',
                        help="Path to trained RandomForest model")
    parser.add_argument('-i','--interval', type=int, default=100,
                        help="Nominal symbol interval in ms")
    parser.add_argument('-w','--window', type=int, default=30,
                        help="Window size in ms for averaging")
    parser.add_argument('-o','--output', default='decoded.csv',
                        help="Output CSV of decoded labels")
    parser.add_argument('-d','--debug', action='store_true',
                        help="Print per-window details and error report")
    args = parser.parse_args()

    # load trace
    df = load_power_df(args.power)
    base = int(df['time_ms'].min())
    init_total = int((df['time_ms'].max() - base)//args.interval) + 1

    # load ground-truth and trim
    y_full = load_seq(args.seq)
    total = min(init_total, len(y_full))
    if len(y_full) < init_total:
        print(f"Warning: only decoding first {total} windows (ground truth length)")

    # compute starts & features
    starts = compute_symbol_starts(df, base, args.interval, total)
    avgs, centers = window_averages(df, starts, args.window)
    temps = np.interp(centers, df['time_ms'], df['temp_C'])

    prev_avgs  = np.concatenate([[avgs[0]],  avgs[:-1]])
    prev_temps = np.concatenate([[temps[0]], temps[:-1]])

    X = np.column_stack((avgs, temps, prev_avgs, prev_temps))
    valid = ~np.isnan(X).any(axis=1)
    Xv = X[valid]
    starts_v = np.array(starts)[valid]
    y = np.array(y_full[:total])[valid]

    # load model & predict
    clf = joblib.load(args.model)
    preds = clf.predict(Xv)

    # debug / error report
    if args.debug:
        feature_stack = np.column_stack((avgs[valid], temps[valid], prev_avgs[valid], prev_temps[valid], y))
        mismatches = 0
        print("start_ms, power_avg, temp_C, true_label, pred_label")
        for s, (p, t, pp, pt, true), pred in zip(starts_v, feature_stack, preds):
            print(f"{int(s):6d}, {p:7.3f}, {t:7.2f}, {pp:7.3f}, {pt:7.2f}, {int(true):3d}, {pred:3d}")
            if pred != true:
                mismatches += 1
        err = mismatches/len(y)*100
        print(f"\nError rate: {err:.2f}% ({mismatches}/{len(y)})")
        # summary of mispredictions
        print("\nMispredicted windows:")
        print("start_ms, power_avg, temp_C, prev_power, prev_temp, true_label, pred_label")
        for s, (p, t, pp, pt, true), pred in zip(starts_v, feature_stack, preds):
            if pred != true:
                print(f"{int(s):6d}, {p:7.3f}, {t:7.2f}, {pp:7.3f}, {pt:7.2f}, {int(true):3d}, {pred:3d}")

    # save decoded CSV
    out_df = pd.DataFrame({'start_ms': starts_v, 'pred_label': preds})
    out_df.to_csv(args.output, index=False)
    print(f"Wrote {len(preds)} decoded symbols → {args.output}")

if __name__ == '__main__':
    main()
