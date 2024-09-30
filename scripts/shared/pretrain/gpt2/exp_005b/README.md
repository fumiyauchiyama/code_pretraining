# fineweb
CUDA_VISIBLE_DEVICES=0 bash scripts/shared/pretrain/gpt2/exp_005b/fineweb.sh > outputs/logs/20240605_exp5b_fineweb.log 2>&1

# python
CUDA_VISIBLE_DEVICES=1 bash scripts/shared/pretrain/gpt2/exp_005b/python.sh > outputs/logs/20240605_exp5b_python.log 2>&1
