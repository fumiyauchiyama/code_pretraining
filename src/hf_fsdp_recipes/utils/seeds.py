from transformers import set_seed, enable_full_determinism

def set_seed(seed: int) -> None:
    # set seed for random, numpy and torch
    set_seed(seed, deterministic=False)
    enable_full_determinism(seed)