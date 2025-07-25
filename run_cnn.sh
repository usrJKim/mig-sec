#!/bin/bash
# run ./run_cnn.sh

#====Configuration======
PROBER_MIG_UUID="MIG_GPU-xxxxx"
MODEL_MIG_UUID="MIG_GPU-xxxxx"

MODE="train"
BATCH=8
EPOCH=10

LOOP=1 # Repeat multiple times to compensate for noise
for ((j=1; j<=$LOOP; j++)); do
#======================= 
  echo "=======${j}/${LOOP} iteration========"
  for MODEL in "resnet" "vgg19" "alexnet" "densenet" "mobilenet"; do
    MODEL_TYPE="${MODEL}"
    MODEL_ARG="--mode ${MODE} --model ${MODEL} --batch ${BATCH} --epochs ${EPOCH}"
      
    TIMESTAMP=$(date +"%m%d_%H%M")
    PROBER_OUT="${MODEL_TYPE}_${MODE}_B${BATCH}_E${EPOCH}_${TIMESTAMP}_power.csv"
    
    # Run prober
    sudo docker run --rm --name prober-container \
    --gpus "device=${PROBER_MIG_UUID}" \
    -v $(pwd)/outputs:/outputs \
    prober-image /outputs/prober/${PROBER_OUT} &
    
    PROBER_PID=$!
    
    sleep 1 # to allow the prober to start first
    
    # Run model
    sudo docker run -it --rm \
    --gpus "device=${MODEL_MIG_UUID}" \
    cnn-image $MODEL_ARG
  
    sudo docker stop prober-container
    sleep 1 # to reduce differences between models
  done
done
