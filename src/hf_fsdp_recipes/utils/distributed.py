import torch.distributed as dist
import os

def set_mpi_env() -> None:
    global_rank = int(os.getenv("OMPI_COMM_WORLD_RANK", 0))
    local_rank = int(os.getenv("OMPI_COMM_WORLD_LOCAL_RANK", 0))
    world_size = int(os.getenv("OMPI_COMM_WORLD_SIZE", 1))

    os.environ["RANK"] = str(global_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)

def setup(rank, world_size):
    # initialize the process group
    dist.init_process_group("nccl", world_size=world_size, rank=rank)

def clean_up():
    dist.destroy_process_group()