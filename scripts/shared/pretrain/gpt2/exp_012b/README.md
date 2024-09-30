exp009について、データのシャッフルと分割のシードを42から1にした。
バッジサイズも16から24にした。

CUDA_VISIBLE_DEVICES=0 bash /home/uchiyama.fumiya/ucllm/ucllm/program/scripts/shared/pretrain/gpt2/exp_012b/python_cf_r.sh > /home/uchiyama.fumiya/ucllm/ucllm/program/scripts/shared/pretrain/gpt2/exp_012b/python_cf_r.out 2>&1

CUDA_VISIBLE_DEVICES=1 bash /home/uchiyama.fumiya/ucllm/ucllm/program/scripts/shared/pretrain/gpt2/exp_012b/python_cf_rwd.sh > /home/uchiyama.fumiya/ucllm/ucllm/program/scripts/shared/pretrain/gpt2/exp_012b/python_cf_rwd.out 2>&1

CUDA_VISIBLE_DEVICES=2 bash /home/uchiyama.fumiya/ucllm/ucllm/program/scripts/shared/pretrain/gpt2/exp_012b/python_cf.sh > /home/uchiyama.fumiya/ucllm/ucllm/program/scripts/shared/pretrain/gpt2/exp_012b/python_cf.out 2>&1