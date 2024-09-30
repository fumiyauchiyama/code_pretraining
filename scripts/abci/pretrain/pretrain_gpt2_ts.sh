#!/bin/bash
#$ -l rt_F=1
#$ -l h_rt=20:00:00
#$ -l USE_SSH=1
#$ -j y
#$ -N pretrain_gpt2_ts
#$ -o logs/pretrain/gpt2/ts/
#$ -cwd

source /etc/profile.d/modules.sh
module load python/3.11 cuda/11.7 cudnn/8.6 nccl/2.12 hpcx/2.12

export PYTHON_SCRIPT=src/pretrain/pretrain.py
export APP_DIR=/groups/gcf51099/uchiyama.fumiya/ucllm/ucllm/program

export HF_HOME=$APP_DIR/.hf_cache
export HF_DATASETS_CACHE=/scratch/$(whoami)/.hf_datasets_cache

cd $APP_DIR
source .venv/bin/activate

python3 \
  $PYTHON_SCRIPT \
  dataset=thestack_ts
