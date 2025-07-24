#!/bin/bash
source myenv/bin/activate
for MODEL in "resnet" "alexnet" "vgg19" "mobilenet" "densenet"; do
  python ./visualize.py --model $MODEL
done
deactivate
