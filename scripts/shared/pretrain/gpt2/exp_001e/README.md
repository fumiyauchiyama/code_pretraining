# fineweb
97657 * 2048
bFloat16 enabled for mixed precision - using bfSixteen policy  
UserWarning: FSDP is switching to use `NO_SHARD` instead of ShardingStrategy.FULL
_SHARD since the world size is 1.
num_iters: 9156, lr_warmup_steps: 915
ucllm/program/outputs/models/openai-community/gpt2/HuggingFaceFW/fineweb/job_1716257056/checkpoint-3

# haskell
97657 * 2048
num_iters: 9156, lr_warmup_steps: 915
ucllm/program/outputs/models/openai-community/gpt2/bigcode/the-stack/job_1716296608/checkpoint-3

# wikipedia
97657 * 2048
ucllm/program/outputs/models/openai-community/gpt2/wikipedia/job_1716258577/checkpoint-3


## haskell
lm_eval --model hf \
    --model_args pretrained=/home/uchiyama.fumiya/ucllm/ucllm/program/outputs/models/openai-community/gpt2/bigcode/the-stack/job_1716296608/checkpoint-3 \
    --tasks fld \
    --num_fewshot 3 \
    --device cuda:1 \
    --log_samples \
    --output_path results/exp1e \
    --batch_size 64
hf (pretrained=/home/uchiyama.fumiya/ucllm/ucllm/program/outputs/models/openai-community/gpt2/bigcode/the-stack/job_1716296608/checkpoint-3), gen_kwargs: (None), limit: None, num_fewshot: 3, batch_size: 64
|    Tasks     |Version|     Filter      |n-shot|  Metric   |Value|   |Stderr|
|--------------|-------|-----------------|-----:|-----------|----:|---|-----:|
|fld           |N/A    |remove_whitespace|     3|exact_match|    0|±  |     0|
| - fld_default|      2|remove_whitespace|     3|exact_match|    0|±  |     0|
| - fld_star   |      2|remove_whitespace|     3|exact_match|    0|±  |     0|

|Groups|Version|     Filter      |n-shot|  Metric   |Value|   |Stderr|
|------|-------|-----------------|-----:|-----------|----:|---|-----:|
|fld   |N/A    |remove_whitespace|     3|exact_match|    0|±  |     0|

## fineweb
lm_eval --model hf \
    --model_args pretrained=/home/uchiyama.fumiya/ucllm/ucllm/program/outputs/models/openai-community/gpt2/HuggingFaceFW/fineweb/job_1716257056/checkpoint-3 \
    --tasks fld \
    --num_fewshot 3 \
    --device cuda:2 \
    --log_samples \
    --output_path results/exp1e \
    --batch_size 64
hf (pretrained=/home/uchiyama.fumiya/ucllm/ucllm/program/outputs/models/openai-community/gpt2/HuggingFaceFW/fineweb/job_1716257056/checkpoint-3), gen_kwargs: (None), limit: None, num_fewshot: 3, batch_size: 64
|    Tasks     |Version|     Filter      |n-shot|  Metric   |Value|   |Stderr|
|--------------|-------|-----------------|-----:|-----------|----:|---|-----:|
|fld           |N/A    |remove_whitespace|     3|exact_match|    0|±  |     0|
| - fld_default|      2|remove_whitespace|     3|exact_match|    0|±  |     0|
| - fld_star   |      2|remove_whitespace|     3|exact_match|    0|±  |     0|

|Groups|Version|     Filter      |n-shot|  Metric   |Value|   |Stderr|
|------|-------|-----------------|-----:|-----------|----:|---|-----:|
|fld   |N/A    |remove_whitespace|     3|exact_match|    0|±  |     0|

# wikipedia
lm_eval --model hf \
    --model_args pretrained=/home/uchiyama.fumiya/ucllm/ucllm/program/outputs/models/openai-community/gpt2/wikipedia/job_1716258577/checkpoint-3 \
    --tasks fld \
    --num_fewshot 3 \
    --device cuda:2 \
    --log_samples \
    --output_path results/exp1e \
    --batch_size 64

hf (pretrained=/home/uchiyama.fumiya/ucllm/ucllm/program/outputs/models/openai-community/gpt2/wikipedia/job_1716258577/checkpoint-3), gen_kwargs: (None), limit: None, num_fewshot: 3, batch_size: 64
|    Tasks     |Version|     Filter      |n-shot|  Metric   |Value|   |Stderr|
|--------------|-------|-----------------|-----:|-----------|----:|---|-----:|
|fld           |N/A    |remove_whitespace|     3|exact_match|    0|±  |     0|
| - fld_default|      2|remove_whitespace|     3|exact_match|    0|±  |     0|
| - fld_star   |      2|remove_whitespace|     3|exact_match|    0|±  |     0|

|Groups|Version|     Filter      |n-shot|  Metric   |Value|   |Stderr|
|------|-------|-----------------|-----:|-----------|----:|---|-----:|
|fld   |N/A    |remove_whitespace|     3|exact_match|    0|±  |     0|