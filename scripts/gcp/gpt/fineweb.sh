#!/bin/bash

# Command line options go here
#SBATCH --partition=a3
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --job-name=outputs-gpt2xl_2048-fineweb-1
#SBATCH --output=outputs-gpt2xl_2048-fineweb-1.out
#SBATCH --gpus-per-node=8
#SBATCH --nodelist=nodelist=slurm0-a3-ghpc-[0-3]

# Command(s) goes here
source ~/miniconda3/etc/profile.d/conda.sh
conda activate myenv

export HF_DATASETS_CACHE="/storage8/fumiyau/.cache/hf_datasets_cache"
export HF_HOME="/storage8/fumiyau/.cache/hf_home"

NUM_GPUS=1
NUM_GPU_PER_NODE=$NUM_GPUS
HOSTFILE_NAME=hostfile/hostfile
MASTER_ADDR="127.0.0.1"
MASTER_PORT=12353

## Total number of GPUs.
num_gpus_pernode=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Number of GPUs per node: $num_gpus_pernode"
num_node="${SLURM_JOB_NUM_NODES}"

num_gpus=$((${num_gpus_pernode} * ${num_node}))

# Sets the master port number to a unique number.
MASTER_PORT=$((10000 + (${SLURM_JOB_ID} % 50000)))

# Creates a hostfile.
script_dir=$(dirname "$0")
hostfile="${script_dir}/hostfile_jobid-${SLURM_JOB_ID}"
nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST)

for node in $nodes
do
  gpu_count=$(ssh ${node} "nvidia-smi --query-gpu=name --format=csv,noheader | wc -l")
  echo "${node} slots=${gpu_count}"
  #ssh $node "source ~/.bashrc"
  #ssh $node 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate .venv_train'
done > "${hostfile}"

echo "hostfile = ${hostfile}"
cat ${hostfile}
echo ""

mpirun -np $NUM_GPUS \
  --npernode $num_gpus_pernode \
  -hostfile $hostfile \
  -x MASTER_PORT=$MASTER_PORT \
  python3 src/hf_fsdp_recipes/pretrain.py \
  model=gpt2xl_2048 \
  dataset=fineweb_100b \
  preprocess=preprocess_exp5 \
  train=train_shared train.batch_size=28 train.test_batch_size=28 \
  train.output_dir="/storage9/fumiyau/ucllm/program/outputs/models/exp1g/fineweb_10b" \
  exec.use_fp16=false

conda deactivate