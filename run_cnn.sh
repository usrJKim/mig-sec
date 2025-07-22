#!/bin/bash
# run ./run.sh model_configuration

#====Configuration======
PROBER_MIG_UUID="MIG_GPU-xxxxx"
MODEL_MIG_UUID="MIG_GPU-xxxxx"
ITER_TIME=1 # repeat running model ITER_TIME times
for ((j=1; j<=100; j++)); do
#======================= 
  echo "=======${j}/100 iteration========"
  for MODEL in "resnet" "vgg19" "alexnet" "densenet" "mobilenet"; do
    MODEL_TYPE="${MODEL}"
    MODEL_ARG="--model ${MODEL}"
      
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    PROBER_OUT="${MODEL_TYPE}_${TIMESTAMP}_power.csv"
    
    # Run prober
    sudo docker run --rm --name prober-container \
    --gpus "device=${PROBER_MIG_UUID}" \
    -v $(pwd)/outputs:/outputs \
    prober-image /outputs/prober/${PROBER_OUT} &
    
    PROBER_PID=$!
    
    sleep 1 # wait to run prober first
    
    # Run model
    for ((i=1; i<=ITER_TIME; i++)); do
      sudo docker run -it --rm \
      --gpus "device=${MODEL_MIG_UUID}" \
      model-image $MODEL_ARG
    done
  
    sudo docker stop prober-container
    sleep 1
  done
done
