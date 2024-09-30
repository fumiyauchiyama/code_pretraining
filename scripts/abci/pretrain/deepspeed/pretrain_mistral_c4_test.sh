#!/bin/bash
#$ -l rt_F=8
#$ -l h_rt=0:30:00
#$ -l USE_SSH=1
#$ -j y
#$ -N pretrain_mistral_c4_deepspeed_test
#$ -o logs/pretrain/mistral_deepspeed/c4/
#$ -cwd

source /etc/profile.d/modules.sh
module load python/3.11 cuda/11.8/11.8.0 cudnn/8.8/8.8.1 nccl/2.16/2.16.2-1 hpcx/2.12

export PYTHON_SCRIPT=src/pretrain/pretrain.py
export APP_DIR=/groups/gcf51099/uchiyama.fumiya/ucllm/ucllm/program

export HF_HOME=$APP_DIR/.hf_cache
export HF_DATASETS_CACHE=/scratch/$(whoami)/.hf_datasets_cache

cd $APP_DIR
source .venv_ds/bin/activate

# distributed settings
source scripts/abci/pretrain/deepspeed/init_distributed.sh

# run deepspeed-integrated hf trainer from MPI launcher

# -mca btl openib,vader,self \
# -mca btl_openib_if_include mlx5_0,mlx5_1,mlx5_2,mlx5_3 \
# -mca coll hcoll \
# -mca mca_btl_openib_use_message_coalescing 1 \
# -mca hwloc_base_binding_policy core \
# -mca mpi_show_mca_params all \
# -mca mpi_warn_on_fork 0 \
mpirun -np $NUM_GPUS \
  --npernode $NUM_GPU_PER_NODE \
  -hostfile $HOSTFILE_NAME \
  -x MASTER_ADDR=$MASTER_ADDR \
  -x MASTER_PORT=$MASTER_PORT \
  -bind-to none -map-by ppr:${NUM_GPU_PER_NODE}:node \
  -x NCCL_DEBUG=INFO -x PATH \
  -mca pml ob1 -mca btl self,tcp \
  -mca coll ^hcoll \
  --mca btl_tcp_if_include bond0 \
  python3 $PYTHON_SCRIPT \
  model=mistral_base \
  dataset=c4_base \
  exec=exec_ds3 \
  train=train_ds_base \
  ++preprocess.preprocess=["select","shuffle","avoid_short_sample","train_test_split"]
