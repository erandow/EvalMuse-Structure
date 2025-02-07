#!/bin/bash

# Change directory to the target location
# cd /mnt/bn/rahf/mlx/users/jincheng.liang/repo/12094/RAHF || exit

# Print the environment variable to verify it is set
# echo "PYTORCH_CUDA_ALLOC_CONF is set to $PYTORCH_CUDA_ALLOC_CONF"

# Install Python dependencies from requirements.txt
pip3 install -r requirements.txt

# Force reinstall numpy to a specific version
pip3 install --force-reinstall numpy==1.25.2

# Set environment variable for CUDA memory allocation
# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Run the Python script
# python3 train_public_cluster.py
python3 -m torch.distributed.launch \
    --nproc_per_node=8 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="localhost" \
    --master_port=12345 \
    train.py \
    --experiment_name "xxx" \
    --lr 2e-5 \
    --iters 2000 \
    --batch_size 4 \
    --accumulate_step 8 \
    --val_iter 50 \
    --save_iter 100 \
    --warmup \
    --data_path xxx \
