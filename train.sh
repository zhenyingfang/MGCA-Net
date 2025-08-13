#!/bin/bash

for i in {0..9}; do
    CUDA_VISIBLE_DEVICES=0 torchrun \
        --nnodes=1 \
        --nproc_per_node=1 \
        --rdzv_backend=c10d \
        --rdzv_endpoint=localhost:0 \
        tools/train.py \
        configs/actionformer_ov/thumos_i3d_50_${i}.py
done

for i in {0..9}; do
    CUDA_VISIBLE_DEVICES=0 torchrun \
        --nnodes=1 \
        --nproc_per_node=1 \
        --rdzv_backend=c10d \
        --rdzv_endpoint=localhost:0 \
        tools/train.py \
        configs/actionformer_ov/thumos_i3d_75_${i}.py
done
