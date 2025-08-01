#!/bin/bash
LOWER=2
LINE=1000000
for GPU in "h100" "a100"; do
  for MIG in "mig1g" "mig2g" "mig3g" "mig4g"; do
    rm editted/$GPU/$MIG/*
    searchdir=`ls ./prober/$GPU/$MIG/`
    for entry in $searchdir; do
      LINES=$(wc -l < "./prober/${GPU}/${MIG}/${entry}")
      head -n 1 "./prober/${GPU}/${MIG}/${entry}" > "./editted/${GPU}/${MIG}/${entry}"
      tail -n +"${LOWER}" "./prober/${GPU}/${MIG}/${entry}" | head -n "${LINE}">> "./editted/${GPU}/${MIG}/${entry}"
    done
  done
done
