#!/bin/sh
#$ -l rt_AF=1
#$ -l h_rt=01:00:00
#$ -j y
#$ -o logs/exp005c/gpt2_xl/python/
#$ -cwd

source /etc/profile.d/modules.sh
module load python/3.11 cuda/11.8 cudnn/8.6 nccl/2.16 hpcx-mt/2.12
source .venv/bin/activate

source scripts/abci_fsdp/init_distributed.sh

export HF_HOME=/scratch/$(whoami)/.cache/huggingface
export HF_DATASETS_CACHE=/scratch/$(whoami)/.hf_datasets_cache

mpirun -np $NUM_GPUS \
  --npernode $NUM_GPU_PER_NODE \
  -hostfile $HOSTFILE_NAME \
  -x MASTER_ADDR=$MASTER_ADDR \
  -x MASTER_PORT=$MASTER_PORT \
  -bind-to none -map-by slot \
  -x PATH \
  python3 src/hf_fsdp_recipes/pretrain.py \
  model=gpt2xl_2048 \
  dataset=thestack_python_all \
  preprocess=preprocess_exp5c \
  train=train_shared train.batch_size=24 train.test_batch_size=24 \
  train.output_dir="/home/uchiyama.fumiya/ucllm/ucllm/program/outputs/models/exp5c/thestack_python" \
  exec.use_fp16=false

deactivate
