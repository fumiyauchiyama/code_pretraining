# fineweb
CUDA_VISIBLE_DEVICES=0 bash scripts/shared/pretrain/gpt2/exp_003/shallow.sh > scripts/shared/pretrain/gpt2/exp_003/shallow.log 2>&1

# haskell
CUDA_VISIBLE_DEVICES=1 bash scripts/shared/pretrain/gpt2/exp_003/middle.sh > scripts/shared/pretrain/gpt2/exp_003/middle.log 2>&1

# wikipedia
CUDA_VISIBLE_DEVICES=2 bash scripts/shared/pretrain/gpt2/exp_003/deep.sh > scripts/shared/pretrain/gpt2/exp_003/deep.log 2>&1
