exp009について、データのシャッフルと分割のシードを42から2にした。

CUDA_VISIBLE_DEVICES=0 bash /home/uchiyama.fumiya/ucllm/ucllm/program/scripts/shared/pretrain/gpt2/exp_009c/python_cf_s.sh > /home/uchiyama.fumiya/ucllm/ucllm/program/scripts/shared/pretrain/gpt2/exp_009c/python_cf_s.out 2>&1

CUDA_VISIBLE_DEVICES=2 bash /home/uchiyama.fumiya/ucllm/ucllm/program/scripts/shared/pretrain/gpt2/exp_009c/python_cf.sh > /home/uchiyama.fumiya/ucllm/ucllm/program/scripts/shared/pretrain/gpt2/exp_009c/python_cf.out 2>&1

CUDA_VISIBLE_DEVICES=1 bash /home/uchiyama.fumiya/ucllm/ucllm/program/scripts/shared/pretrain/gpt2/exp_009c/python_raw.sh > /home/uchiyama.fumiya/ucllm/ucllm/program/scripts/shared/pretrain/gpt2/exp_009c/python_raw.out 2>&1