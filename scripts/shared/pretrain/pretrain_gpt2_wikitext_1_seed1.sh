
export PYTHON_SCRIPT=src/pretrain/pretrain.py
export LOG_DIR=logs/pretrain_gpt2_wikitext_1
export LOG_NAME=pretrain_gpt2_wikitext_001_seed1.txt

source environments/env_vars.sh
mkdir -p $LOG_DIR

CUDA_VISIBLE_DEVICES=1 python3 \
  $PYTHON_SCRIPT \
  --base_model "openai-community/gpt2" \
  --dirname $OUTPUT_DIR \
  --dataset_name "wikitext" \
  --num_total_samples 50000 \
  --num_train_epochs 5 \
  --min_token_length 1000 \
  --seed 1 \
  > $LOG_DIR/$LOG_NAME

