# fineweb
CUDA_VISIBLE_DEVICES=0 bash scripts/shared/pretrain/gpt2/exp_001f/fineweb.sh

# haskell
CUDA_VISIBLE_DEVICES=1 bash scripts/shared/pretrain/gpt2/exp_001f/haskell.sh

# wikipedia
CUDA_VISIBLE_DEVICES=2 bash scripts/shared/pretrain/gpt2/exp_001f/wikipedia.sh





# fineweb
CUDA_VISIBLE_DEVICES=0 bash scripts/shared/pretrain/gpt2/exp_001g/fineweb3.sh > scripts/shared/pretrain/gpt2/exp_001g/fineweb3.txt 2>&1

# haskell
CUDA_VISIBLE_DEVICES=1 bash scripts/shared/pretrain/gpt2/exp_001g/haskell3.sh > scripts/shared/pretrain/gpt2/exp_001g/haskell3.txt 2>&1