import pandas as pd
import matplotlib.pyplot as plt
import glob

def labelling(file):
    if 'resnet' in file:
        return 'ResNet18'
    elif 'vgg' in file:
        return 'VGG19'
    elif 'alexnet' in file:
        return 'AlexNet'
    elif 'densenet' in file:
        return 'DenseNet121'
    elif 'mobilenet' in file:
        return 'MobileNet_V2'
    elif 'mistral' in file:
        return "Mistral-7B-Instruct"
    elif 'tinyllama' in file:
        return 'TinyLlama-1.1B-Chat'
    elif 'phi-2' in file:
        return 'Phi-2'
    elif 'gemma' in file:
        return 'Gemma-2b-it'

csv_files = glob.glob("./prober/*.csv")

plt.figure(figsize=(12,6))

for file in csv_files:
    df = pd.read_csv(file)
    label = labelling(file)
    plt.plot(df['time_ms'], df['power_w'], label=label)

plt.xlabel("Time (msec)")
plt.ylabel('Power (W)')
plt.title('Power Consumption over Time(MIG 1g.10GB)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("summary_plot.png")
