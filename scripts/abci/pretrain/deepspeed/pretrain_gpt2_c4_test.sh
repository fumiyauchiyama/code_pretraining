#!/bin/bash
#$ -l rt_F=2
#$ -l h_rt=20:00:00
#$ -l USE_SSH=1
#$ -j y
#$ -N pretrain_gpt2_c4_deepspeed_test
#$ -o logs/pretrain/gpt2_deepspeed/c4/
#$ -cwd

source /etc/profile.d/modules.sh
module load python/3.11 cuda/11.8/11.8.0 cudnn/8.8/8.8.1 nccl/2.16/2.16.2-1 hpcx/2.12

export PYTHON_SCRIPT=src/pretrain/pretrain.py
export APP_DIR=/groups/gcf51099/uchiyama.fumiya/ucllm/ucllm/program

export HF_HOME=$APP_DIR/.hf_cache
export HF_DATASETS_CACHE=/scratch/$(whoami)/.hf_datasets_cache

cd $APP_DIR
source .venv/bin/activate

# distributed settings
source scripts/abci/pretrain/deepspeed/init_distributed.sh

# run deepspeed-integrated hf trainer from MPI launcher
mpirun -np $NUM_GPUS \
  --npernode $NUM_GPU_PER_NODE \
  -hostfile $HOSTFILE_NAME \
  -x MASTER_ADDR=$MASTER_ADDR \
  -x MASTER_PORT=$MASTER_PORT \
  -bind-to none -map-by slot \
  -x NCCL_DEBUG=INFO -x PATH \
  -mca pml ob1 -mca btl ^openib \
  -mca coll ^hcoll \
  --mca btl_tcp_if_include eno1 \
  python3 $PYTHON_SCRIPT \
  dataset=c4_base \
  exec=exec_ds \
  train=train_ds_base \
  ++preprocess.preprocess=["select","shuffle","avoid_short_sample","train_test_split"]
