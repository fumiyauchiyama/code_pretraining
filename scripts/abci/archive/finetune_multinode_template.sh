#!/bin/bash
#$ -l rt_F=6
#$ -l h_rt=4:00:00
#$ -l USE_SSH=1
#$ -j y
#$ -N finetune_deepspeed_multinode
#$ -o logs/
#$ -cwd

source /etc/profile.d/modules.sh
module load python/3.11 cuda/11.7 cudnn/8.6 nccl/2.12 hpcx/2.12

export PYTHON_SCRIPT=src/finetune/finetune_deepspeed.py
export CONFIG=config/finetune_deepspeed_template.yaml
export APP_DIR=/groups/gcf51099/uchiyama.fumiya/ucllm/program

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export HF_HOME=$APP_DIR/.hf_cache
export HF_DATASETS_CACHE=$APP_DIR/.hf_datasets_cache

cd $APP_DIR
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

hostfile=$(mktemp)
for l in `cat $SGE_JOB_HOSTLIST`; do echo $l slots=4; done > $hostfile
trap "rm $hostfile" EXIT
trap "trap - EXIT; rm $hostifle; exit -1" INT PIPE TERM

MASTER_ADDR=$HOSTNAME deepspeed \
  --master_addr $HOSTNAME \
  --hostfile $hostfile \
  --no_ssh_check \
  --launcher OpenMPI \
  --launcher_args "-mca coll ^hcoll" \
  $PYTHON_SCRIPT \
  --config_file $CONFIG

