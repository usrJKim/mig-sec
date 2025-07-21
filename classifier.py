#!/usr/bin/env python3
import os
import glob
import joblib
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

DATA_DIR = "./data"
OUT_DIR  = "./models"
os.makedirs(OUT_DIR, exist_ok=True)

class ThresholdClassifier:
    """Univariate Gaussian threshold‐based classifier."""
    def __init__(self):
        self.stats = {}  # label -> (mean,var)

    def fit(self, X, y):
        for c in np.unique(y):
            vals = X[y == c].ravel()
            self.stats[int(c)] = (float(vals.mean()), float(vals.var()))

    def predict(self, X):
        X = np.atleast_1d(X).ravel()
        preds = []
        for x in X:
            best_c, best_p = None, -np.inf
            for c, (mu, var) in self.stats.items():
                p = np.exp(-0.5 * ((x - mu) ** 2) / var) / np.sqrt(2 * np.pi * var)
                if p > best_p:
                    best_p, best_c = p, c
            preds.append(best_c)
        return np.array(preds)

def load_data():
    X, y = [], []
    for path in glob.glob(os.path.join(DATA_DIR, "*.csv")):
        label = int(os.path.splitext(os.path.basename(path))[0])
        print(f"Loading data from {path} for label {label}")
        df = pd.read_csv(path)
        X.extend(df["power_w"].values)
        y.extend([label] * len(df))
    return np.array(X).reshape(-1, 1), np.array(y)

def main():
    X, y = load_data()

    # ——— Label‐wise percentile trimming ———
    pct = 1.0  # drop bottom 1% and top 1% per label
    keep = []
    for c in np.unique(y):
        idxs = np.where(y == c)[0]
        vals = X[idxs].ravel()
        low_p, high_p = np.percentile(vals, [pct, 100 - pct])
        good = idxs[(vals >= low_p) & (vals <= high_p)]
        print(f"Label {c}: trimming to [{low_p:.3f}, {high_p:.3f}], "
              f"{len(good)}/{len(idxs)} kept")
        keep.extend(good.tolist())
    keep = np.unique(keep)
    X, y = X[keep], y[keep]
    print(f"After trimming: {len(keep)} total samples remain.")
    # ——————————————————————————————

    # Print min/max per label after trimming
    for c in np.unique(y):
        vals = X[y == c].ravel()
        print(f"Label {c}: min = {vals.min():.6f}, max = {vals.max():.6f}")

    # Train/test split
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, stratify=y, test_size=0.3, random_state=0
    )

    # 1) Threshold‐Gaussian
    thr = ThresholdClassifier()
    thr.fit(Xtr, ytr)
    y_pred = thr.predict(Xte)
    acc_thr = (y_pred == yte).mean()
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

    # Save models
    joblib.dump(gnb, os.path.join(OUT_DIR, "gnb.joblib"))
    joblib.dump(rf,  os.path.join(OUT_DIR, "rf.joblib"))
    # Save threshold stats as JSON
    with open(os.path.join(OUT_DIR, "thr_stats.json"), "w") as f:
        json.dump({str(k): v for k, v in thr.stats.items()}, f, indent=2)

    print(f"\nModels and stats saved under {OUT_DIR}/")

if __name__ == "__main__":
    main()
