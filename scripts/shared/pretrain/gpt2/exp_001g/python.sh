#!/bin/bash

source .venv/bin/activate

export HF_DATASETS_CACHE="/home/uchiyama.fumiya/.cache/hf_datasets_cache"
export HF_HOME="/home/uchiyama.fumiya/.cache/hf_home"

NUM_GPUS=1
NUM_GPU_PER_NODE=$NUM_GPUS
HOSTFILE_NAME=hostfile/hostfile
MASTER_ADDR="127.0.0.1"
MASTER_PORT=12371

mpirun -np $NUM_GPUS \
  --npernode $NUM_GPU_PER_NODE \
  -hostfile $HOSTFILE_NAME \
  -x MASTER_ADDR=$MASTER_ADDR \
  -x MASTER_PORT=$MASTER_PORT \
  -bind-to none -map-by slot \
  -x PATH \
  python3 src/hf_fsdp_recipes/pretrain.py \
  model=gpt2_2048 \
  dataset=thestack_python \
  train=train_shared train.batch_size=24 train.test_batch_size=24 \
  train.output_dir="/home/uchiyama.fumiya/ucllm/ucllm/program/outputs/models/exp1g/thestack_python" \
  exec.use_fp16=false

deactivate
