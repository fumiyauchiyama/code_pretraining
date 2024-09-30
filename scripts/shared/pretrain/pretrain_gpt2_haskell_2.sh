
export PYTHON_SCRIPT=src/pretrain/pretrain.py
export LOG_DIR=logs/pretrain_gpt2_haskell_2
export LOG_NAME=pretrain_gpt2_haskell_002.txt

source environments/env_vars.sh
mkdir -p $LOG_DIR

CUDA_VISIBLE_DEVICES=1 accelerate launch \
  --num_processes 2 \
  $PYTHON_SCRIPT \
  --base_model "openai-community/gpt2" \
  --dirname $OUTPUT_DIR \
  --dataset_name "bigcode/the-stack" \
  --thestack_language "haskell" \
  --num_total_samples 50000 \
  --num_train_epochs 5 \
  --seed 0 \
  > $LOG_DIR/$LOG_NAME

