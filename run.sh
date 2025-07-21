#!/bin/bash
# run ./run.sh model_configuration

#====Configuration======
PROBER_MIG_UUID="MIG_GPU-xxxxx"
MODEL_MIG_UUID="MIG_GPU-xxxxx"
ITER_TIME=3 # repeat running model ITER_TIME times
MODE="train"
MODEL="gpt"
KV="False"
BATCH="8" # If gpt infer, and use kv then, batch should be 1

for LAYERS in 4 8 16; do
  MODEL_ARGS="--mode ${MODE} \
    --model ${MODEL} \
    --usekv ${KV} \
    --epochs 3 \
    --batch_size ${BATCH} \
    --lr 1e-4 \
    --num_heads 4 \
    --num_layers ${LAYERS} \
    --model_dim 64 \
    --seq_len 128 \
    --vocab_size 1000"
#======================= 
  MODEL_TYPE="${MODEL}_${MODE}_Layer${LAYERS}_kv_${KV}"
    
  TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
  PROBER_OUT="${MODEL_TYPE}_${TIMESTAMP}_power.csv"
  
  # Run prober
  docker run --rm \
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
  
    docker run --rm \
    --gpus "device=${MODEL_MIG_UUID}" \
    model-image \
    $MODEL_ARGS > outputs/${MODEL_OUT} 2> outputs/${MODEL_ERR}
  done
  
  kill $PROBER_PID 2>/dev/null
  wait $PROBER_PID 2>/dev/null
done
