#!/bin/bash
RESNET_MIG_UUID="MIG_GPU-xxxxx"
INTERFER_MIG_UUID="MIG_GPU-xxxxx"

# run 50 times alone
echo "Running baseline (ResNet only)..."
for i in {1..50}; do
    echo "Trial $i"
    sudo docker run --rm \
        -v $PWD:/app \
        --gpus "device=${RESNET_MIG_UUID}" \
        resnet-image \
        python3 resnet.py --model resnet152 /app/image/n01768244 > "results/alone_${i}.txt"
done

# run 50 times simulataneously
echo "Running with interference..."
for i in {1..50}; do
    echo "Trial $i"
    # Start interference container (other MIG)
    sudo docker run --rm --name interfer-container \
        --gpus "device=${INTERFER_MIG_UUID}" \
        -d \
        loadgen-image

    sleep 1  # allow load to start

    # Run resnet
    sudo docker run --rm \
        -v $PWD:/app \
        --gpus "device=${INTERFER_MIG_UUID}" \
        resnet-image \
        python3 resnet.py --model resnet152 /app/image/n01768244 > "results/simu_${i}.txt"

    # Docker --rm and background container auto-terminate assumed
    sudo docker stop interfer-container
done

