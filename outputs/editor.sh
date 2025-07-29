#!/bin/bash
for GPU in "h100" "a100"; do
  for MIG in "mig1g" "mig2g" "mig3g" "mig4g"; do
    searchdir=`ls ./prober/$GPU/$MIG/`
    for entry in $searchdir; do
      head -n 50000 "./prober/${GPU}/${MIG}/${entry}" > "./editted/${GPU}/${MIG}/${entry}"
    done
  done
done
