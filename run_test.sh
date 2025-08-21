#!/bin/bash
# run ./run.sh model_configuration

#====Configuration======
PROBER_MIG_UUID="MIG_GPU-xxxxx"
MODEL_MIG_UUID="MIG_GPU-xxxxx"
#=======================

for MIG in 1 2 3 4; do
  PROBER_OUT="MIG${MIG}g_test_IntegerSum.csv"
  # Run prober
  sudo docker run --rm --name prober-container \
    --gpus "device=${PROBER_MIG_UUID}" \
    -v $(pwd)/outputs:/outputs \
    prober-image /outputs/prober/${PROBER_OUT} &

  sleep 0.1 # wait to run prober first

  #RUN test
  sudo docker run -it --rm \
    --gpus "device=${MODEL_MIG_UUID}" \
    test-image 100 100 "input_files/pulse_mig${MIG}g.csv"

  sudo docker stop prober-container
done
