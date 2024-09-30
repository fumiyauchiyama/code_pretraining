#!/bin/bash
#$ -l rt_F=1
#$ -l h_rt=20:00:00
#$ -l USE_SSH=1
#$ -j y
#$ -N pretrain_gpt2_wikitext_1
#$ -o logs/program/gpt2/wikitext/
#$ -cwd

source /etc/profile.d/modules.sh
module load python/3.11 cuda/11.7 cudnn/8.6 nccl/2.12 hpcx/2.12

export PYTHON_SCRIPT=src/pretrain/pretrain_wiki.py
export APP_DIR=/groups/gcf51099/uchiyama.fumiya/ucllm/program

export HF_HOME=$APP_DIR/.hf_cache
export HF_DATASETS_CACHE=/scratch/$(whoami)/.hf_datasets_cache

cd $APP_DIR
source .venv/bin/activate

python3 \
  $PYTHON_SCRIPT \
  --base_model "openai-community/gpt2" \
  --num_total_samples 50000 \
  --num_train_epochs 5 \
