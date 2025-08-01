import pandas as pd
import matplotlib.pyplot as plt
import glob
import argparse

def get_args():
    parser = argparse.ArgumentParser()

    # Input
    parser.add_argument("--gpu", type=str, choices=["h100","a100", "h100_new"])
    parser.add_argument("--mig", type=str, choices=["mig1g","mig2g","mig3g","mig4g"])
    return parser.parse_args()

def labelling(file):
    if 'resnet' in file:
        return 'ResNet18', "#1f77b4"
    elif 'vgg' in file:
        return 'VGG19',"#ff7f0e"
    elif 'alexnet' in file:
        return 'AlexNet', "#2ca02c"
    elif 'densenet' in file:
        return 'DenseNet121',"#d62728"
    elif 'mobilenet' in file:
        return 'MobileNet_V2',"#9467bd" 
    elif 'TinyLlama' in file:
        return 'TinyLlama-1.1B-Chat',"#8c564b"
    elif 'Llama' in file:
        return "Meta-Llama-3-8B-Instruct","#e377c2"
    elif 'Phi-2' in file:
        return 'Phi-2',"#7f7f7f" 
    elif 'Gemma' in file:
        return 'Gemma-2b-it',"#17becf"

parser = get_args()
csv_files = glob.glob(f"./editted/{parser.gpu}/{parser.mig}/*.csv")
plt.figure(figsize=(12,6))

for file in csv_files:
    df = pd.read_csv(file)
    label,color = labelling(file)
    plt.plot(df['time_ms'], df['power_w'], label=label,color=color)

plt.xlabel("Time (msec)")
plt.ylabel('Power (W)')
plt.title(f'Power Consumption over Time({parser.gpu}, {parser.mig})')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"./summary_plot/summary_plot_{parser.gpu}_{parser.mig}.png")
