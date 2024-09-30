#!/bin/sh
#$ -l rt_G.small=1
#$ -l h_rt=01:00:00
#$ -j y
#$ -o logs/fsdp_test/gpt2_base/wikitext/
#$ -cwd

source /etc/profile.d/modules.sh
module load python/3.11 cuda/11.8 cudnn/8.6 nccl/2.16 hpcx-mt/2.12
source .venv/bin/activate

source scripts/abci_fsdp/init_distributed.sh

export HF_HOME=/scratch/$(whoami)/.cache/huggingface
export HF_DATASETS_CACHE=/scratch/$(whoami)/.hf_datasets_cache

# -map-by ppr:${NUM_GPU_PER_NODE}:node \
# -mca pml ob1 -mca btl self,tcp \
# -mca btl_tcp_if_include bond0 \

mpirun -np $NUM_GPUS \
  --npernode $NUM_GPU_PER_NODE \
  -hostfile $HOSTFILE_NAME \
  -x MASTER_ADDR=$MASTER_ADDR \
  -x MASTER_PORT=$MASTER_PORT \
  -bind-to none -map-by slot \
  -x PATH \
  python3 src/hf_fsdp_recipes/pretrain.py \
  model=gpt2_base \
  dataset=wikitext_base

deactivate