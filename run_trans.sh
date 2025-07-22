#!/bin/bash
# run ./run.sh model_configuration

#====Configuration======
PROBER_MIG_UUID="MIG_GPU-xxxxx"
MODEL_MIG_UUID="MIG_GPU-xxxxx"
ITER_TIME=3 # repeat running model ITER_TIME times
MODE="train"
MODEL="gpt"
KV="False"
BATCH="10" # If gpt infer, and use kv then, batch should be 1

for ((j=1;j<=10;j++)); do
  for LAYERS in 4 8 12 16 20 24; do
    MODEL_ARGS="--mode ${MODE} \
      --model ${MODEL} \
      --usekv ${KV} \
      --epochs 10 \
      --batch_size ${BATCH} \
      --lr 1e-4 \
      --num_heads 4 \
      --num_layers ${LAYERS} \
      --model_dim 1024 \
      --seq_len 512 \
      --vocab_size 1024"
  #======================= 
    MODEL_TYPE="${MODEL}_${MODE}_Layer${LAYERS}_kv_${KV}"
      
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
    
      sudo docker run -it --rm \
      --gpus "device=${MODEL_MIG_UUID}" \
      model-image $MODEL_ARGS 
    done
  
    sudo docker stop prober-container
    sleep 1
  done
done
