#!/bin/bash
#$ -l rt_AG.small=1
#$ -l h_rt=24:00:00
#$ -j y
#$ -N finetune_fld_gpt2_haskell_500000
#$ -o logs/
#$ -cwd

source /etc/profile.d/modules.sh
module load python/3.11 cuda/11.7 cudnn/8.6 nccl/2.12 hpcx/2.12

export PYTHON_SCRIPT=src/finetune/finetune.py
export APP_DIR=/groups/gcf51099/uchiyama.fumiya/ucllm/program/src/FLD-prover

export HF_HOME=$APP_DIR/../../.hf_cache
export HF_DATASETS_CACHE=/scratch/$(whoami)/.hf_datasets_cache

cd $APP_DIR
source .venv/bin/activate
export PYTHONPATH=`pwd -P`:$PYTHONPATH

python ./scripts/run_causal_prover.py  \
    --FLD_dataset_name hitachi-nlp/FLD.v2 \
    --FLD_dataset_config_name default \
    --model_name_or_path /scratch/acc13097es/models/ucllm_program1709107463_haskell_500000/checkpoint-6000 \
    --tokenizer_name openai-community/gpt2 \
    --output_dir /scratch/$(whoami)/models/ucllm_program_fld/ \
    --logging_dir outputs/tensorboard/ \
    --seed 0  \
    --max_grad_norm 0.5   \
    --max_steps 70  \
    --gradient_accumulation_steps 4  \
    --max_eval_samples 150  \
    --learning_rate 1e-05  \
    --warmup_steps 21  \
    --max_target_length 1000  \
    --logging_strategy steps  \
    --logging_steps 1  \
    --overwrite_output_dir True  \
    --no_subproof_for_unknown True  \
    --per_device_train_batch_size 1  \
    --per_device_eval_batch_size 1  \
    --dataloader_num_workers 0  \
    --log_examples True  \
    --max_train_samples 5  \
    --FLD_dataset_prob 1.0  \
    --FLD_max_eval_samples 150  \
    --eval_steps 70  \
    --remove_unused_columns False  \
    --instruction False  \
    --streaming False    \
    --evaluation_strategy steps  \
    --save_strategy no  \
    --save_model_at_end False  \
    --gradient_checkpointing True  \
    --block_size 1000  \
    --FLD_proof_eval_padding longest   \
    --generation_do_sample False  \
    --generation_temperature 1.0     \
    --generation_timeout 7200  \
    --evaluation_timeout 36000  \
    --do_train True  \
    --do_eval_in_outerloop False  \
    --do_predict False  \
    --fp16 True  \
    --lr_scheduler_type linear  \
    --weight_decay 0.0  \
    --lora False  \
    --use_auth_token

