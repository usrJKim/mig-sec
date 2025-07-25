#!/bin/bash
# run ./run_trans.sh 

#====Configuration======
PROBER_MIG_UUID="MIG_GPU-xxxxx"
MODEL_MIG_UUID="MIG_GPU-xxxxx"
ITER_TIME=3 # repeat running model ITER_TIME times
LOOP=1

for ((j=1;j<=$LOOP;j++)); do
  for MODEL in "Mistral" "TinyLlama" "Phi-2" "Gemma"; do
    MODEL_ARGS="--model ${MODEL}"
  #======================= 
    MODEL_TYPE="${MODEL}"
      
    TIMESTAMP=$(date +"%m%d_%H%M")
    PROBER_OUT="${MODEL_TYPE}_${TIMESTAMP}.csv"
    
    # Run prober
    sudo docker run --rm --name prober-container\
    --gpus "device=${PROBER_MIG_UUID}" \
    -v $(pwd)/outputs:/outputs \
    prober-image /outputs/prober/${PROBER_OUT} &
    
    PROBER_PID=$!
    
    sleep 1 # wait to run prober first
    
    # Run model
      sudo docker run -it --rm \
      --gpus "device=${MODEL_MIG_UUID}" \
      llm-image $MODEL_ARGS 
  
    sudo docker stop prober-container
    sleep 1
  done
done
