#!/bin/bash

CUDA_VISIBLE_DEVICES=4 LR="2e-4" RUN_NAME="moe_lr2e-4" nohup taskset -c 64-128 python run_moe.py &> moe_lr2e-4.log &
CUDA_VISIBLE_DEVICES=5 LR="5e-4" RUN_NAME="moe_lr5e-4" nohup taskset -c 64-128 python run_moe.py &> moe_lr5e-4.log &
CUDA_VISIBLE_DEVICES=6 LR="1e-3" RUN_NAME="moe_lr1e-3" nohup taskset -c 64-128 python run_moe.py &> moe_lr1e-3.log &
CUDA_VISIBLE_DEVICES=7 LR="2e-3" RUN_NAME="moe_lr2e-3" nohup taskset -c 64-128 python run_moe.py &> moe_lr2e-3.log &
CUDA_VISIBLE_DEVICES=4 LR="5e-3" RUN_NAME="moe_lr5e-3" nohup taskset -c 64-128 python run_moe.py &> moe_lr5e-3.log &
CUDA_VISIBLE_DEVICES=5 LR="1e-2" RUN_NAME="moe_lr1e-2" nohup taskset -c 64-128 python run_moe.py &> moe_lr1e-2.log &

CUDA_VISIBLE_DEVICES=4 LR="2e-4" RUN_NAME="dense_lr2e-4" nohup taskset -c 64-128 python run_dense.py &> dense_lr2e-4.log &
CUDA_VISIBLE_DEVICES=5 LR="5e-4" RUN_NAME="dense_lr5e-4" nohup taskset -c 64-128 python run_dense.py &> dense_lr5e-4.log &
CUDA_VISIBLE_DEVICES=6 LR="1e-3" RUN_NAME="dense_lr1e-3" nohup taskset -c 64-128 python run_dense.py &> dense_lr1e-3.log &
CUDA_VISIBLE_DEVICES=7 LR="2e-3" RUN_NAME="dense_lr2e-3" nohup taskset -c 64-128 python run_dense.py &> dense_lr2e-3.log &
CUDA_VISIBLE_DEVICES=6 LR="5e-3" RUN_NAME="dense_lr5e-3" nohup taskset -c 64-128 python run_dense.py &> dense_lr5e-3.log &
CUDA_VISIBLE_DEVICES=7 LR="1e-2" RUN_NAME="dense_lr1e-2" nohup taskset -c 64-128 python run_dense.py &> dense_lr1e-2.log &