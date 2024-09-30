#!/bin/bash
#$ -l rt_F=1
#$ -l h_rt=0:20:00
#$ -l USE_SSH=1
#$ -j y
#$ -N pretrain_gemma2b_c4
#$ -o logs/pretrain/gemma2b/c4/
#$ -cwd

source /etc/profile.d/modules.sh
module load python/3.11 cuda/11.8/11.8.0 cudnn/8.8/8.8.1 nccl/2.16/2.16.2-1 hpcx/2.12

export PYTHON_SCRIPT=src/pretrain/pretrain.py
export APP_DIR=/groups/gcf51099/uchiyama.fumiya/ucllm/ucllm/program

export HF_HOME=$APP_DIR/.hf_cache
export HF_DATASETS_CACHE=/scratch/$(whoami)/.hf_datasets_cache

cd $APP_DIR
source .venv/bin/activate

python3 \
  $PYTHON_SCRIPT \
  dataset=c4_base \
  model=gemma2b_base \
  ++preprocess.preprocess=["select","shuffle","avoid_short_sample","train_test_split"]
