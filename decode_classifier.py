#!/usr/bin/env python3
import sys
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

class ThresholdClassifier:
    def __init__(self, stats):
        self.stats = {int(k):tuple(v) for k,v in stats.items()}
    def predict(self, X):
        X = np.atleast_1d(X).ravel()
        preds = []
        for x in X:
            best_c, best_p = None, -np.inf
            for c,(mu,var) in self.stats.items():
                p = np.exp(-0.5*((x-mu)**2)/var) / np.sqrt(2*np.pi*var)
                if p > best_p:
                    best_p, best_c = p, c
            preds.append(best_c)
        return np.array(preds)

def main():
    if len(sys.argv) < 4:
        print(f"Usage: {sys.argv[0]} <model_type> <model_path> <input.csv> [output.csv]")
        print("  model_type: thr | gnb | rf")
        sys.exit(1)

    model_type = sys.argv[1]
    model_path = sys.argv[2]
    infile     = sys.argv[3]
    outfile    = sys.argv[4] if len(sys.argv)>4 else "decoded.csv"

    df = pd.read_csv(infile)  # expects columns ID,Power
    X = df["Power"].values.reshape(-1,1)

    # load model
    if model_type == "thr":
        stats = json.load(open(model_path))
        clf = ThresholdClassifier(stats)
    elif model_type == "gnb":
        clf = joblib.load(model_path)
    elif model_type == "rf":
        clf = joblib.load(model_path)
    else:
        raise ValueError("Unknown model_type")

    y_pred = clf.predict(X)
    df["Predicted"] = y_pred
    df.to_csv(outfile, index=False)
    print(f"Wrote decoded symbols to {outfile}")

if __name__=="__main__":
    main()
