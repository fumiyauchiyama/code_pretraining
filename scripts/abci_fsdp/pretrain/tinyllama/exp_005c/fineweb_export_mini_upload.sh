#!/bin/sh
#$ -l rt_C.large=1
#$ -l h_rt=24:00:00
#$ -j y
#$ -o logs/exp005c/data_export_mini_upload/fineweb-tinyllama/
#$ -cwd

source /etc/profile.d/modules.sh
module load python/3.11 cuda/11.8 cudnn/8.6 nccl/2.16 hpcx-mt/2.12
source .venv/bin/activate

export HF_HOME=/scratch/$(whoami)/.cache/huggingface
export HF_DATASETS_CACHE=/scratch/$(whoami)/.hf_datasets_cache

python3 src/hf_fsdp_recipes/tools/upload_dataset.py \
  --path /groups/gcf51099/uchiyama.fumiya/ucllm/ucllm/program/outputs/data/sample-100BTTinyLlama/TinyLlama_v1.11000000 \
  --name fumiyau/utllm-program-fineweb-1m-tinyllama

deactivate