#!/bin/bash
# run ./run.sh model_configuration

#====Configuration======
PROBER_MIG_UUID="MIG_GPU-xxxxx"
MODEL_MIG_UUID="MIG_GPU-xxxxx"
MIG=4
#=======================
for PULSE in 100 90 80 70 60 50 40 30 20 15 10; do
  for TYPE in sum fmul sfu gemv shared; do
    for var in {1..1}; do
      PROBER_OUT="MIG${MIG}g_pulse${PULSE}_${TYPE}_${var}.csv"

      # Run prober
      sudo docker run --rm --name prober-container \
        --gpus "device=${PROBER_MIG_UUID}" \
        -v "$(pwd)/outputs:/outputs" \
        prober-image "/outputs/prober/${PROBER_OUT}" &

      sleep 1 # wait to run prober first

      #RUN test
      sudo docker run -it --rm \
        --gpus "device=${MODEL_MIG_UUID}" \
        test-image "./test_${TYPE}" $PULSE $PULSE "input_files/pulse_mig${MIG}g.csv"

      sudo docker stop prober-container
      sleep 1
    done
  done
done
