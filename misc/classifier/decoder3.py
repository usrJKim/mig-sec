#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.naive_bayes     import GaussianNB
from sklearn.ensemble         import RandomForestClassifier
from sklearn.neighbors        import KNeighborsClassifier
from sklearn.svm              import SVC
from sklearn.linear_model     import LogisticRegression

def load_sequence(seq_csv, n_calib):
    with open(seq_csv, 'r') as f:
        text = f.read()
    tokens = text.replace('\n', ',').split(',')
    seq = [int(tok) for tok in tokens if tok.strip()]
    if len(seq) < n_calib:
        raise ValueError(f"Need {n_calib} labels, got {len(seq)}")
    return np.array(seq[:n_calib], dtype=int)

def load_power_df(power_csv):
    df = pd.read_csv(power_csv)
    if "time_ms" not in df or "power_w" not in df:
        raise ValueError("power CSV must have columns time_ms,power_w")
    return df.set_index("time_ms", drop=False)

def first_non_outlier(vals):
    q1, q3 = np.percentile(vals, [25,75])
    iqr    = q3 - q1
    low, high = q1 - 1.5*iqr, q3 + 1.5*iqr
    for v in vals:
        if low <= v <= high:
            return v
    return np.median(vals)

def sample_windows(df, interval_ms, slack_ms, n_windows, t0=None):
    if t0 is None:
        t0 = int(df["time_ms"].min())
    picks, centers = [], []
    for i in range(n_windows):
        w0 = t0 + i*interval_ms
        w1 = w0 + slack_ms
        window = df.loc[(df["time_ms"]>=w0)&(df["time_ms"]<=w1), "power_w"].values
        pick = first_non_outlier(window) if window.size else df.iloc[df.index.get_indexer([w0], method="nearest")[0]]["power_w"]
        picks.append(pick)
        centers.append(w0 + slack_ms//2)
    return np.array(picks), np.array(centers)

def get_classifier(name):
    if name=="threshold": return None
    if name=="gnb":       return GaussianNB()
    if name=="rf":        return RandomForestClassifier(n_estimators=200, random_state=0)
    if name=="knn":       return KNeighborsClassifier(n_neighbors=5)
    if name=="svc":       return SVC(kernel="rbf", probability=True, random_state=0)
    if name=="logreg":    return LogisticRegression(max_iter=500, random_state=0)
    raise ValueError(f"Unknown model {name}")

def build_bigram(seq, K, alpha=1.0):
    counts = np.zeros((K,K))
    for i in range(len(seq)-1):
        counts[seq[i], seq[i+1]] += 1
    counts += alpha
    P = counts / counts.sum(axis=1, keepdims=True)
    return np.log(P)

def viterbi(log_obs, log_trans):
    T,K = log_obs.shape
    delta = np.full((T,K), -np.inf)
    psi   = np.zeros((T,K),dtype=int)
    delta[0] = log_obs[0]
    for t in range(1,T):
        for j in range(K):
            scores = delta[t-1] + log_trans[:,j]
            psi[t,j] = np.argmax(scores)
            delta[t,j] = scores[psi[t,j]] + log_obs[t,j]
    path = np.zeros(T,dtype=int)
    path[-1] = np.argmax(delta[-1])
    for t in range(T-1,0,-1):
        path[t-1] = psi[t, path[t]]
    return path

def main():
    p = argparse.ArgumentParser(description="Bigram-aware power decoder with calibration plot")
    p.add_argument("-p","--power",   default="power_data.csv")
    p.add_argument("-s","--sequence",default="./input_files/sequence.csv")
    p.add_argument("-i","--interval",type=int, default=100)
    p.add_argument("-l","--slack",   type=int, default=10)
    p.add_argument("-n","--nc",      type=int, default=150)
    p.add_argument("-m","--model",
        choices=["threshold","gnb","rf","knn","svc","logreg"], default="gnb")
    p.add_argument("-o","--output",  default="decoded_bigram.csv")
    args = p.parse_args()

    # 1) Load data & calibration preamble
    df = load_power_df(args.power)
    cal_picks, cal_centers = sample_windows(df, args.interval, args.slack, args.nc)
    seq = load_sequence(args.sequence, args.nc)
    K = len(np.unique(seq))

    # 2) Calibration plot
    plt.figure(figsize=(12,4))
    plt.plot(cal_centers, cal_picks, '-o', ms=4)
    for t,pw,lbl in zip(cal_centers,cal_picks,seq):
        plt.text(t, pw+0.2, str(lbl), color='blue', ha='center', va='bottom', fontsize=8)
        if lbl==0:
            plt.text(t, pw-0.5, f"{int(t)} ms", color='red', ha='center', va='top', fontsize=7)
    plt.xlabel("Time (ms)")
    plt.ylabel("Power (W)")
    plt.title(f"Calibration Preamble (first {args.nc} windows of {args.interval}±{args.slack} ms)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("calibration_plot.png", dpi=300)
    print("Saved calibration_plot.png")

    # 3) Bigram & classifier training
    log_trans = build_bigram(seq, K, alpha=1.0)
    clf = get_classifier(args.model)
    if args.model=="threshold":
        stats = {lbl:(cal_picks[seq==lbl].mean(), cal_picks[seq==lbl].var()) for lbl in range(K)}
    else:
        clf.fit(cal_picks.reshape(-1,1), seq)

    # 4) Sample all windows
    total_w = int(((df.time_ms.max()-df.time_ms.min())//args.interval)+1)
    all_picks, all_centers = sample_windows(df, args.interval, args.slack, total_w)

    # 5) Observation log‐likelihoods
    T = len(all_picks)
    log_obs = np.zeros((T,K))
    if args.model=="threshold":
        for lbl,(mu,var) in stats.items():
            log_obs[:,lbl] = norm.logpdf(all_picks, loc=mu, scale=np.sqrt(var))
    else:
        log_obs = clf.predict_log_proba(all_picks.reshape(-1,1))

    # 6) Viterbi decode
    path = viterbi(log_obs, log_trans)

    # 7) Save post‐calibration decoded
    decoded = [
        (int(all_centers[i]), float(all_picks[i]), int(path[i]))
        for i in range(args.nc, T)
    ]
    out_df = pd.DataFrame(decoded, columns=["time_ms","power_w","decoded_label"])
    out_df.to_csv(args.output, index=False)
    print(f"Decoded {len(decoded)} symbols → '{args.output}'")

if __name__=="__main__":
    main()
