#!/bin/bash
# run ./run.sh model_configuration

#====Configuration======
PROBER_MIG_UUID="MIG_GPU-xxxxx"
MODEL_MIG_UUID="MIG_GPU-xxxxx"
#======================= 
PROBER_OUT="test_power.csv"

# Run prober
sudo docker run --rm --name prober-container \
--gpus "device=${PROBER_MIG_UUID}" \
-v $(pwd)/outputs:/outputs \
prober-image /outputs/prober/${PROBER_OUT} &

PROBER_PID=$!

sleep 1 # wait to run prober first

#RUN test
  sudo docker run -it --rm \
  --gpus "device=${MODEL_MIG_UUID}" \
  test-image
done

sudo docker stop prober-container
