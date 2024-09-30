# fineweb
CUDA_VISIBLE_DEVICES=0 bash scripts/shared/pretrain/gpt2/exp_005b/fineweb.sh > outputs/logs/20240605_exp5b_fineweb.log 2>&1

# python
CUDA_VISIBLE_DEVICES=1 bash scripts/shared/pretrain/gpt2/exp_005b/python.sh > outputs/logs/20240605_exp5b_python.log 2>&1


[acc13097es@es1 ucllm]$ show_point
Group                 Disk            CloudStorage                    Used           Point   Used%
gag51404                30                  0.0000            127,177.7971         281,000      45
  `- acc13097es          -                       -                  109.08               -       0
gcc50626                 1                  0.0000                728.3023           1,000      73
  `- acc13097es          -                       -                  0.0000               -       0
gcd50654                40                  0.0000             15,672.9977          17,000      92
  `- acc13097es          -                       -                  0.0000               -       0
gcf51099                13                142.8627              8,859.5446          10,000      89
  `- acc13097es          -                       -                429.2259               -       4