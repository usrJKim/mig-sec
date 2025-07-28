#!/bin/bash
sudo apt install git-lfs

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$SCRIPT_DIR"

git clone https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1
git clone https://huggingface.co/google/gemma-2b-it
git clone https://huggingface.co/microsoft/phi-2
git clone https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0
