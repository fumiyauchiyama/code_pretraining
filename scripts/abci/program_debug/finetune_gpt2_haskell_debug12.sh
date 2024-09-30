#!/bin/bash
#$ -l rt_F=1
#$ -l h_rt=15:00:00
#$ -l USE_SSH=1
#$ -j y
#$ -N finetune_gpt2_haskell_debug12
#$ -o logs/
#$ -cwd

source /etc/profile.d/modules.sh
module load python/3.11 cuda/11.7 cudnn/8.6 nccl/2.12 hpcx/2.12

export PYTHON_SCRIPT=src/finetune/finetune_customdata.py
export APP_DIR=/groups/gcf51099/uchiyama.fumiya/ucllm/program

export HF_HOME=$APP_DIR/.hf_cache
export HF_DATASETS_CACHE=/scratch/$(whoami)/.hf_datasets_cache

cd $APP_DIR
source .venv/bin/activate

CUDA_VISIBLE_DEVICES=1 python3 \
  $PYTHON_SCRIPT \
  --base_model "openai-community/gpt2" \
  --dataset_name "wikitext" \
  --dataset_subname "wikitext-103-v1" \
  --num_total_samples 1100000 \
  --lr_scheduler_type "constant_with_warmup" \
  --seed 1

