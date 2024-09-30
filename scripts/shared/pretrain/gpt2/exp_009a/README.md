exp009について、データのシャッフルと分割のシードを42から1にした。
バッジサイズも16から24にした。

CUDA_VISIBLE_DEVICES=2 bash /home/uchiyama.fumiya/ucllm/ucllm/program/scripts/shared/pretrain/gpt2/exp_009a/python_cf_s.sh > /home/uchiyama.fumiya/ucllm/ucllm/program/scripts/shared/pretrain/gpt2/exp_009a/python_cf_s.out 2>&1

[in progress]
CUDA_VISIBLE_DEVICES=2 bash /home/uchiyama.fumiya/ucllm/ucllm/program/scripts/shared/pretrain/gpt2/exp_009a/python_cf.sh > /home/uchiyama.fumiya/ucllm/ucllm/program/scripts/shared/pretrain/gpt2/exp_009a/python_cf.out 2>&1

[in progress]
CUDA_VISIBLE_DEVICES=1 bash /home/uchiyama.fumiya/ucllm/ucllm/program/scripts/shared/pretrain/gpt2/exp_009a/python_raw.sh > /home/uchiyama.fumiya/ucllm/ucllm/program/scripts/shared/pretrain/gpt2/exp_009a/python_raw.out 2>&1