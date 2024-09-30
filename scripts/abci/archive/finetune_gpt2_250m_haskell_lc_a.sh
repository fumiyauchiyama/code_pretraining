#!/bin/bash
#$ -l rt_AF=1
#$ -l h_rt=72:00:00
#$ -l USE_SSH=1
#$ -j y
#$ -N finetune_gpt2_250m_haskell_lc_a
#$ -o logs/
#$ -cwd

source /etc/profile.d/modules.sh
module load python/3.11 cuda/11.7 cudnn/8.6 nccl/2.12 hpcx/2.12

export PYTHON_SCRIPT=src/finetune/finetune.py
export CONFIG=config/finetune_gpt2_250m_haskell_lc.yaml
export APP_DIR=/groups/gcf51099/uchiyama.fumiya/ucllm/program

export HF_HOME=$APP_DIR/.hf_cache
export HF_DATASETS_CACHE=/scratch/$(whoami)/.hf_datasets_cache

cd $APP_DIR
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

python3 \
  $PYTHON_SCRIPT \
  --config_file $CONFIG

