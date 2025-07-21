#!/usr/bin/env python3
import os
import csv
import joblib
import json
import pandas as pd
import numpy as np
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

# ——— CONFIGURATION ———
POWER_CSV    = "power_data.csv"        # your millisecond trace
SEQUENCE_CSV = "./input_files/sequence.csv"
MODEL_DIR    = "./models"
INTERVAL_MS  = 100                     # sampling every 100 ms
# ———————————————————

os.makedirs(MODEL_DIR, exist_ok=True)

class ThresholdClassifier:
    """Univariate Gaussian threshold‐based classifier."""
    def __init__(self):
        self.stats = {}  # label -> (mean, var)

    def fit(self, X, y):
        for c in np.unique(y):
            vals = X[y==c].ravel()
            self.stats[int(c)] = (float(vals.mean()), float(vals.var()))

    def predict(self, X):
        X = np.atleast_1d(X).ravel()
        preds = []
        for x in X:
            best_c, best_p = None, -np.inf
            for c, (mu, var) in self.stats.items():
                p = np.exp(-0.5*((x-mu)**2)/var) / np.sqrt(2*np.pi*var)
                if p > best_p:
                    best_p, best_c = p, c
            preds.append(best_c)
        return np.array(preds)

def load_sequence(seq_csv):
    """Read sequence.csv (10 symbols per line) and flatten to [0..14] labels."""
    seq = []
    with open(seq_csv) as f:
        reader = csv.reader(f)
        for row in reader:
            seq.extend(int(x) for x in row if x.strip()!='')
    return np.array(seq, dtype=int)

def sample_power(power_csv, interval_ms):
    """Load time_ms,power_w and sample every `interval_ms` ms (nearest)."""
    df = pd.read_csv(power_csv)
    df = df.set_index("time_ms")
    t_max = int(df.index.max())
    times = np.arange(0, t_max+1, interval_ms)
    samples = []
    for t in times:
        if t in df.index:
            samples.append(df.at[t, "power_w"])
        else:
            # fallback to nearest
            idx = df.index.get_indexer([t], method="nearest")[0]
            samples.append(df.iloc[idx]["power_w"])
    return np.array(samples)

def make_dataset():
    # 1) load true labels
    labels = load_sequence(SEQUENCE_CSV)
    # 2) load power samples
    powers = sample_power(POWER_CSV, INTERVAL_MS)
    # 3) truncate to same length
    n = min(len(labels), len(powers))
    X = powers[:n].reshape(-1,1)        # feature matrix (n_samples,1)
    y = labels[:n]                      # label vector
    return X, y

def main():
    X, y = make_dataset()
    print(f"Dataset: {len(X)} samples, {len(np.unique(y))} classes")

    # split
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, stratify=y, test_size=0.3, random_state=0
    )

    # 1) Threshold‐Gaussian
    thr = ThresholdClassifier()
    thr.fit(Xtr, ytr)
    acc_thr = (thr.predict(Xte) == yte).mean()
    print(f"Threshold‐Gaussian accuracy: {acc_thr:.3f}")

    # 2) GaussianNB
    gnb = GaussianNB().fit(Xtr, ytr)
    acc_gnb = gnb.score(Xte, yte)
    print(f"GaussianNB accuracy:           {acc_gnb:.3f}")

    # 3) RandomForest
    rf = RandomForestClassifier(n_estimators=250, random_state=0)
    rf.fit(Xtr, ytr)
    acc_rf = rf.score(Xte, yte)
    print(f"RandomForest accuracy:         {acc_rf:.3f}")

    # save
    joblib.dump(gnb, os.path.join(MODEL_DIR, "gnb.joblib"))
    joblib.dump(rf,  os.path.join(MODEL_DIR, "rf.joblib"))
    with open(os.path.join(MODEL_DIR, "thr_stats.json"), "w") as f:
        json.dump({str(k):v for k,v in thr.stats.items()}, f, indent=2)
    print(f"Models saved under {MODEL_DIR}/")

if __name__=="__main__":
    main()
