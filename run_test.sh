#!/bin/bash
# run ./run.sh model_configuration

#====Configuration======
PROBER_MIG_UUID="MIG_GPU-xxxxx"
MODEL_MIG_UUID="MIG_GPU-xxxxx"
MIG=1
BIT=1
#======================= 

for ITER in 1 2 3 4 5 6 7 8 9 10; do
  PROBER_OUT="MIG${MIG}g_test_${ITER}.csv"
  # Run prober
  sudo docker run --rm --name prober-container \
  --gpus "device=${PROBER_MIG_UUID}" \
  -v $(pwd)/outputs:/outputs \
  prober-image /outputs/prober/${PROBER_OUT} &

  PROBER_PID=$!

  sleep 0.1 # wait to run prober first

  #RUN test
  sudo docker run -it --rm \
  --gpus "device=${MODEL_MIG_UUID}" \
  test-image "input_files/mig${MIG}g/labels${BIT}bit.csv"
  
  sudo docker stop prober-container
done
