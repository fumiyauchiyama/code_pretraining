#!/bin/sh
#$ -l rt_C.small=1
#$ -l h_rt=40:00:00
#$ -j y
#$ -o logs/download_data/fineweb/
#$ -cwd

source /etc/profile.d/modules.sh
module load python/3.11 cuda/11.8 cudnn/8.6 nccl/2.16
source .venv/bin/activate

export HF_HOME=/scratch/$(whoami)/.cache/huggingface
export HF_DATASETS_CACHE=/scratch/$(whoami)/.hf_datasets_cache

python3 src/hf_fsdp_recipes/tools/download_datasets.py \
  --dataset_name slimpajama

deactivate