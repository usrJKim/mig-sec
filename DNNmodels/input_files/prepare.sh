#!/bin/bash
for MIG in 1 2 3 4; do
  OUTPUT_DIR="./mig${MIG}g"
  INPUT="./bin.txt"
  NUM_SM=$((MIG*14))
  python bin_to_label.py -i $INPUT -o $OUTPUT_DIR --num_sm $NUM_SM
done 
