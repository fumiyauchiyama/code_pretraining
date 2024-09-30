#!/bin/bash

source .venv/bin/activate

export HF_DATASETS_CACHE="/home/uchiyama.fumiya/.cache/hf_datasets_cache"
export HF_HOME="/home/uchiyama.fumiya/.cache/hf_home"

NUM_GPUS=1
NUM_GPU_PER_NODE=$NUM_GPUS
HOSTFILE_NAME=hostfile/hostfile
MASTER_ADDR="127.0.0.1"
MASTER_PORT=12353

mpirun -np $NUM_GPUS \
  --npernode $NUM_GPU_PER_NODE \
  -hostfile $HOSTFILE_NAME \
  -x MASTER_ADDR=$MASTER_ADDR \
  -x MASTER_PORT=$MASTER_PORT \
  -bind-to none -map-by slot \
  -x PATH \
  python3 src/hf_fsdp_recipes/pretrain.py \
  model=gpt2_base \
  dataset=fineweb_10b \
  train=train_shared preprocess.num_total_tokens=20000000 train.batch_size=64 train.test_batch_size=64 \
  train.output_dir="/home/uchiyama.fumiya/ucllm/ucllm/program/outputs/models/exp1c"

deactivate