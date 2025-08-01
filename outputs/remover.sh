#!/bin/bash

for GPU in "a100" "h100"; do
  for MIG in "mig1g" "mig2g" "mig3g" "mig4g"; do
    find editted/$GPU/$MIG/ ! -name '*power.csv' -type f -delete
  done
done
