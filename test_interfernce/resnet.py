#!/usr/bin/env python3
# python3 resnet.py --model resnet152 ./image/n01768244
import argparse
import re
import time
from pathlib import Path

import torch
from torchvision import models
from PIL import Image
import requests
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}

def load_model_and_weights(name: str, device: torch.device):
    m = re.match(r"resnet(\d+)$", name)
    if not m:
        raise ValueError(f"Unsupported model variant: {name}")
    num = m.group(1)
    weights_enum = getattr(models, f"ResNet{num}_Weights", None)
    if weights_enum is None:
        raise ValueError(f"Couldn't find ResNet{num}_Weights in torchvision.models")
    weights = weights_enum.DEFAULT

    model = getattr(models, name)(weights=weights)
    model.eval().to(device)
    return model, weights

def preprocess_image(img_path: Path, weights):
    # we time inside to break down where the cost is
    preprocess = weights.transforms()
    s = time.time()
    img = Image.open(img_path).convert("RGB")
    elapsed1 = time.time() - s

    s = time.time()
    ret = preprocess(img).unsqueeze(0)
    elapsed2 = time.time() - s

    print(f"Image '{img_path.name}' preprocessed in {elapsed1:.6f}s (loading) + {elapsed2:.6f}s (transforming) = {elapsed1 + elapsed2:.6f}s")
    return ret

def load_labels():
    url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
    resp = requests.get(url)
    resp.raise_for_status()
    class_idx = resp.json()
    return {int(k): v[1] for k, v in class_idx.items()}

def run_inference(model, input_tensor, device, topk: int):
    input_tensor = input_tensor.to(device)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.time()
    with torch.no_grad():
        logits = model(input_tensor)
    elapsed = time.time() - start_time

    probs = torch.nn.functional.softmax(logits[0], dim=0)
    top_probs, top_idxs = probs.topk(topk)
    return top_probs.cpu().tolist(), top_idxs.cpu().tolist(), elapsed

def gather_images(input_path: Path):
    if input_path.is_file():
        return [input_path] if input_path.suffix.lower() in IMG_EXTS else []
    elif input_path.is_dir():
        return sorted(
            p for p in input_path.iterdir()
            if p.is_file() and p.suffix.lower() in IMG_EXTS
        )
    else:
        raise ValueError(f"Path not found: {input_path}")

def main():
    parser = argparse.ArgumentParser(description="ResNet batch inference")
    parser.add_argument("input", type=Path,
                        help="Path to an image or a directory of images")
    parser.add_argument("--model", type=str, default="resnet50",
                        choices=["resnet18","resnet34","resnet50","resnet101","resnet152"],
                        help="Which ResNet variant to load")
    parser.add_argument("--topk", type=int, default=5,
                        help="Number of top predictions to show")
    args = parser.parse_args()

    device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, weights = load_model_and_weights(args.model, device)
    labels = load_labels()

    images = gather_images(args.input)
    if not images:
        print(f"No JPEG/PNG/BMP/GIF images found in {args.input}")
        return

    # 1) Preprocess all images first
    t_preprocess = 0.0
    prepped = []
    samples = images[:10]

    for img_path in samples:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        tensor = preprocess_image(img_path, weights)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.time()
        t_preprocess += (t1 - t0)
        prepped.append((img_path, tensor))

    # 2) Run inference one at a time on the preprocessed tensors
    t_infer = 0.0
    for img_path, tensor in prepped:
        probs, idxs, elapsed = run_inference(model, tensor, device, args.topk)
        t_infer += elapsed
        # if you want to print per-image results:
        # print(f"\nTop {args.topk} for '{img_path.name}':")
        # for prob, idx in zip(probs, idxs):
        #     print(f"  {labels[idx]:<25} {prob*100:>6.2f}%")

    # Final summary
    print(f"\nTotal preprocessing time for {len(samples)} image(s): {t_preprocess:.2f} seconds.")
    print(f"Total inference time for {len(samples)} image(s): {t_infer:.2f} seconds.\n")

if __name__ == "__main__":
    main()
