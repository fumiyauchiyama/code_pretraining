# holds various wrapping policies for fsdp


import torch.distributed as dist
import torch.nn as nn
import torch

from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullyShardedDataParallel as FSDP,
    CPUOffload,
    BackwardPrefetch,
    MixedPrecision,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)

import functools
from typing import Type

from hf_fsdp_recipes import utils


def get_size_policy(min_params=1e8):
    num_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=int(min_params)
    )
    return num_wrap_policy


def get_model_wrapper(model_name: str):
    """we register our main layer class and use the fsdp transformer wrapping policy
    ensures embedding layers are in the root fsdp unit for shared access and that fsdp units map to transformer layers
    """
    # ====   use new transformer wrapper

    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            utils.get_model_decoder_layer(model_name),
        },
    )

    return auto_wrap_policy
