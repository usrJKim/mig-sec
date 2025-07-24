import pandas as pd
import math
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
from functools import reduce
import argparse
import random
import gc
from collections import defaultdict

def get_args():
    parser = argparse.ArgumentParser()

    # Input
    parser.add_argument("--model", type=str, choices=["resnet", "vgg19", "alexnet", "densenet", "mobilenet"])

    return parser.parse_args()


args = get_args()
num_samples = 100
bin_cap = 50

model = args.model
print(f"==========MODEL:{model}==========")
# sample csv lists
csv_files = sorted(glob.glob(f"./cnn/{model}_*.csv"))
csv_files = random.sample(csv_files, num_samples)
power_data= []

# 1st loop, calculate mean
print("First loop: Getting mean")
sum_dict = defaultdict(float)
count_dict = defaultdict(int)

for file in tqdm(csv_files):
    df = pd.read_csv(file)[['time_ms', 'power_w']]
    for row in df.itertuples(index=False):
        sum_dict[row.time_ms] += row.power_w
        count_dict[row.time_ms] += 1
    del df
    gc.collect()

# get mean
mean_dict = {t: sum_dict[t] / count_dict[t] for t in sum_dict}

# 2nd loop: standard deviation
print("Second loop: getting std")
squared_diff_sum  = defaultdict(float)
for file in tqdm(csv_files):
    df = pd.read_csv(file)[['time_ms', 'power_w']]
    for row in df.itertuples(index=False):
        diff = row.power_w - mean_dict[row.time_ms]
        squared_diff_sum[row.time_ms] += diff*diff
    del df
    gc.collect()

# get std
std_dict = {
    t: math.sqrt(squared_diff_sum[t]/count_dict[t])
    for t in squared_diff_sum
}

# Visualize
summary_df = pd.DataFrame({
                              'time_ms': list(mean_dict.keys()),
                              'mean_power': list(mean_dict.values()),
                              'std_power': [std_dict[t] for t in mean_dict]
                          }).sort_values('time_ms')
#Time binning
summary_df['time_binned']=(summary_df['time_ms']//bin_cap)*bin_cap
binned_summary = summary_df.groupby('time_binned').agg({
                                                           'mean_power': 'mean',
                                                           'std_power': 'mean'
                                                       }).reset_index()

#Graph
plt.figure(figsize=(14, 6))
plt.errorbar(
    summary_df['time_binned'], 
    summary_df['mean_power'], 
    yerr=summary_df['std_power'], 
    fmt='-o', 
    ecolor='gray', 
    capsize=3,
    label='Mean Power ± Std'
)
plt.title(f'Average Power with Error Bars ({bin_cap}ms bins, {model})')
plt.xlabel('Time (ms)')
plt.ylabel('Power (W)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(f"./plots/{model}_mean_power_errorbar.png", dpi=300)
plt.close()
print("===========DONE==========")

