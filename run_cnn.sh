#!/bin/bash
# run ./run.sh model_configuration

#====Configuration======
PROBER_MIG_UUID="MIG_GPU-xxxxx"
MODEL_MIG_UUID="MIG_GPU-xxxxx"
ITER_TIME=3 # repeat running model ITER_TIME times
#======================= 

for MODEL in "resnet" "vgg19" "alexnet" "densenet" "mobilenet"; do
  MODEL_TYPE="${MODEL}"
    
  TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
  PROBER_OUT="${MODEL_TYPE}_${TIMESTAMP}_power.csv"
  
  # Run prober
  sudo docker run --rm --name prober-container\
  --gpus "device=${PROBER_MIG_UUID}" \
  -v $(pwd)/outputs:/outputs \
  prober-image /outputs/prober/${PROBER_OUT} &
  
  PROBER_PID=$!
  
  sleep 1 # wait to run prober first
  
  # Run model
  for ((i=1; i<=ITER_TIME; i++)); do
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    MODEL_OUT="${MODEL_TYPE}_${TIMESTAMP}_model.log"
    MODEL_ERR="${MODEL_TYPE}_${TIMESTAMP}_model.err"
  
    sudo docker run --rm \
    --gpus "device=${MODEL_MIG_UUID}" \
    model-image \
    $MODEL_ARGS > outputs/${MODEL_OUT} 2> outputs/${MODEL_ERR}
  done

  sudo docker stop prober-container
done
