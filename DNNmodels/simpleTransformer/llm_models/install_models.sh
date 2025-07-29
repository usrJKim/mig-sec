#!/bin/bash
sudo apt install git-lfs
git lfs install

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$SCRIPT_DIR"

git clone https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
git clone https://huggingface.co/google/gemma-2b-it
git clone https://huggingface.co/microsoft/phi-2
git clone https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0
