#!/bin/bash

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME=$APP_DIR/.hf_cache
export HF_DATASETS_CACHE=$APP_DIR/.hf_datasets_cache

cd $APP_DIR

hostfile=$(mktemp)
for l in `cat $SGE_JOB_HOSTLIST`; do echo $l slots=4; done > $hostfile
trap "rm $hostfile" EXIT
trap "trap - EXIT; rm $hostifle; exit -1" INT PIPE TERM

cat $hostfile

MASTER_ADDR=$HOSTNAME poetry run deepspeed \
  --master_addr $HOSTNAME \
  --hostfile $hostfile \
  --no_ssh_check \
  --launcher OpenMPI \
  --launcher_args "-mca coll ^hcoll" \
  $PYTHON_SCRIPT \
  --config_file $CONFIG