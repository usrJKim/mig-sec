#!/bin/bash
for GPU in "h100" "a100"
do
  for MIG in "mig1g" "mig2g" "mig3g" "mig4g"
  do
    python3 plot_gather.py --gpu $GPU --mig $MIG
  done
done
