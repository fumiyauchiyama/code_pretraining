#!/bin/bash
#$ -l rt_C.large=1
#$ -l h_rt=24:00:00
#$ -j y
#$ -N iter_stack_v2_contents
#$ -o logs/utils/property/the_stack_v2/
#$ -cwd

source /etc/profile.d/modules.sh
module load python/3.11 cuda/11.7 cudnn/8.6 nccl/2.12 hpcx/2.12

export PYTHON_SCRIPT=src/utils/iter_stack_v2_contents.py
export APP_DIR=/groups/gcf51099/uchiyama.fumiya/ucllm/program

export HF_HOME=$APP_DIR/.hf_cache
export HF_DATASETS_CACHE=/scratch/$(whoami)/.hf_datasets_cache

cd $APP_DIR
source .venv/bin/activate

python3 \
  $PYTHON_SCRIPT

