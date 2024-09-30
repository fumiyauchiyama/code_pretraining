from hf_fsdp_recipes import policies
from hf_fsdp_recipes.utils.environment import bfloat_support

def get_policies(
    model_name: str,
    mixed_precision: bool,
    use_fp16: bool, 
    rank: int
    ):

    """establish current policies for mixed precision and fsdp wrapping"""

    mixed_precision_policy = None
    wrapping_policy = None

    # mixed precision -----
    if mixed_precision:
        bfloat_available = bfloat_support()
        if bfloat_available and not use_fp16:
            mixed_precision_policy = policies.bfSixteen
            if rank == 0:
                print(f"bFloat16 enabled for mixed precision - using bfSixteen policy")
        elif use_fp16:
            mixed_precision_policy = policies.fpSixteen
            if rank == 0:
                print(f"FP16 enabled. ")
        else:
            # mixed_precision_policy = policies.fpSixteen
            print(
                f"bFloat16 support not present. Will use FP32, and not mixed precision"
            )

    wrapping_policy = policies.get_model_wrapper(model_name)

    return mixed_precision_policy, wrapping_policy
