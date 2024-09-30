#!/bin/sh
#$ -l rt_C.large=1
#$ -l h_rt=72:00:00
#$ -j y
#$ -o logs/exp005c/data_export/python/
#$ -cwd

source /etc/profile.d/modules.sh
module load python/3.11 cuda/11.8 cudnn/8.6 nccl/2.16 hpcx-mt/2.12
source .venv/bin/activate

export HF_HOME=/scratch/$(whoami)/.cache/huggingface
export HF_DATASETS_CACHE=/scratch/$(whoami)/.hf_datasets_cache

python3 src/hf_fsdp_recipes/tools/export_dataset.py \
  --dataset-path bigcode/the-stack \
  --dataset-data-dir data/python \
  --verification-mode no_checks \
  --content-column-name content \
  --pack-texts

deactivate