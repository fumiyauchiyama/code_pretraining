from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import instantiate
from datasets import load_dataset, DatasetDict

import torch
import os
import sys
import time
import logging
import wandb
import math

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    CPUOffload,
    StateDictType,
)
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType

current_path: str = os.getcwd()
sys.path.append(f"{current_path}/src")
sys.path.append(current_path)

from hf_fsdp_recipes.utils import (
    set_mpi_env,
    setup,
    clean_up,
    setup_model,
    get_date_of_run,
    train,
    validation,
    WarmupCosineAnnealingLR,
)

from hf_fsdp_recipes import policies
from hf_fsdp_recipes import model_checkpointing

from hf_fsdp_recipes.dataset import (
    ConcatDataset,
    limit_sample_num,
    avoid_short_samples,
    insert_special_token,
    tokenize_with_padding,
    tokenize,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:

    set_mpi_env()

    local_rank = int(os.environ['LOCAL_RANK'])
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])  

    setup(rank=rank, world_size=world_size) 

    if rank==0:
        logger.info(OmegaConf.to_yaml(cfg))

    if "custom_model_config" in cfg.model.keys():
        custom_model_config = cfg.model.custom_model_config
    else:
        custom_model_config = None
    
    if "custom_tokenizer_config" in cfg.model.keys():
        custom_tokenizer_config = cfg.model.custom_tokenizer_config
    else:
        custom_tokenizer_config = None

    model, tokenizer = setup_model(
        cfg.model.model_name, 
        rank, 
        custom_model_config=custom_model_config,
        custom_tokenizer_config=custom_tokenizer_config,
        )

    if rank==0:
        model_size = sum(t.numel() for t in model.parameters())
        logger.info(f"Model size: {model_size/(1000**2):.1f}M parameters")

    # Set a training output dir and a checkpoint name
    job_name = os.getenv("JOB_NAME", "job")
    job_id = os.getenv("JOB_ID", str(time.time())[:10])
    checkpoint_name = f"{job_name}_{job_id}"

    path = ""

    if "path" in cfg.dataset.load_dataset.keys():
        path = cfg.dataset.load_dataset.path
    else:
        path = cfg.dataset.load_dataset.dataset_path

    output_path = os.path.join(
        cfg.train.output_dir, 
        cfg.model.model_name,
        path,
        checkpoint_name
        )
    
    if rank==0:
        logger.info(f"Training output path: {output_path}")

    if rank==0:
        run = wandb.init(
                project="ucllm-program",
                name=checkpoint_name
            )

    ## save your trained model checkpoint to wandb
    os.environ["WANDB_LOG_MODEL"]="false"
    ## turn off watch to log faster
    os.environ["WANDB_WATCH"]="false"
    
    # Load dataset
    raw_datasets = instantiate(cfg.dataset.load_dataset)

    if "skip" not in cfg.preprocess.preprocess:

        if "path" not in cfg.dataset.load_dataset.keys():
            raw_datasets = raw_datasets.select(range(500000))
        
        if cfg.preprocess.preprocess is not None:
            for idx, i in enumerate(cfg.preprocess.preprocess):

                if rank > 0:
                    logger.info("Waiting for main process to perform the mapping")
                    dist.barrier()
                else:
                    logger.info(f"preprocess: {i}")

                if i == "avoid_short_sample":
                    raw_datasets = raw_datasets.filter(lambda example: len(example[cfg.dataset.content_column_name]) > 0)
                    # if "shuffle" in cfg.preprocess.preprocess[idx]:
                    #     avoid_short_samples(
                    #         raw_datasets,
                    #         cfg.preprocess.select.num_total_samples,
                    #         cfg.preprocess.avoid_short_sample.min_token_length,
                    #         cfg.dataset.content_column_name,
                    #         tokenizer,
                    #         is_shuffled=True,
                    #         seed=cfg.preprocess.shuffle.seed,
                    #     )
                    # else:
                    #     avoid_short_samples(
                    #         raw_datasets,
                    #         cfg.preprocess.select.num_total_samples,
                    #         cfg.preprocess.avoid_short_sample.min_token_length,
                    #         cfg.dataset.content_column_name,
                    #         tokenizer,
                    #         is_shuffled=False,
                    #         seed=cfg.preprocess.shuffle.seed,
                    #     )
                elif i == "shuffle":
                    raw_datasets = raw_datasets.shuffle(
                        seed=cfg.preprocess.shuffle.seed
                        )
                elif i == "select":
                    if cfg.preprocess.select.num_total_samples <= len(raw_datasets):
                        f"num_total_samples is {cfg.preprocess.select.num_total_samples}. However, there are only {len(raw_datasets)} samples. Skipping select process"
                    else:
                        raw_datasets = raw_datasets.select(
                            range(cfg.preprocess.select.num_total_samples)
                            )
                elif i == "prepend_eos":
                    def prepend_eos(sample):
                        sample[cfg.dataset.content_column_name] = tokenizer.eos_token + sample[cfg.dataset.content_column_name]
                        return sample

                    raw_datasets = raw_datasets.map(
                        prepend_eos,
                        num_proc=10,
                        )
                elif i == "train_test_split":
                    raw_datasets = raw_datasets.train_test_split(
                        **cfg.preprocess.train_test_split
                        )
                else :
                    pass

                if rank == 0:
                    print("Loading results from main process")
                    torch.distributed.barrier()

        assert "train" in raw_datasets.keys() \
            and "test" in raw_datasets.keys()
        
        train_dataset = raw_datasets["train"]
        val_dataset = raw_datasets["test"]

        if "pack_texts" in cfg.preprocess.preprocess:
            add_label = lambda sample: \
                {"labels": sample["input_ids"].copy()}
            # ConcatDataset converts to torch.tensor
            train_dataset = tokenize(
                train_dataset, 
                tokenizer,
                cfg.dataset.content_column_name,
                )
            train_dataset = train_dataset.map(
                add_label,
                batched=True, 
                batch_size=2000,
                num_proc=10,
                )
            train_dataset = ConcatDataset(
                train_dataset,
                model.config.max_position_embeddings
                )
            val_dataset = tokenize(
                val_dataset, 
                tokenizer,
                cfg.dataset.content_column_name,
                )
            val_dataset = val_dataset.map(
                add_label,
                batched=True, 
                batch_size=2000,
                num_proc=10,
                )
            val_dataset = ConcatDataset(
                val_dataset,
                model.config.max_position_embeddings,
                )
        else:
            add_label = lambda sample: \
                {"labels": sample["input_ids"].copy()}
            
            train_dataset = tokenize_with_padding(
                train_dataset,
                tokenizer,
                cfg.dataset.content_column_name,
                model.config.max_position_embeddings,
                )
            print(train_dataset[0])
            train_dataset = train_dataset.map(
                add_label,
                num_proc=10,
                )
            val_dataset = tokenize_with_padding(
                val_dataset,
                tokenizer,
                cfg.dataset.content_column_name,
                model.config.max_position_embeddings,
                )
            val_dataset = val_dataset.map(
                add_label,
                num_proc=10,
                )
            
    else:
        print("loading preprocessed dataset")
        train_dataset = raw_datasets["train"]
        val_dataset = raw_datasets["validation"]
        print("preprocessed dataset loaded")
        train_dataset = ConcatDataset(
            train_dataset,
            model.config.max_position_embeddings,
            preprocess=False
            )
        val_dataset = ConcatDataset(
            val_dataset,
            model.config.max_position_embeddings,
            preprocess=False
            )
        print("Successfully converted into ConcatDataset.")

    if cfg.preprocess.num_total_tokens is not None:
        print(f"trying adjusting tokens...")
        num_datasets_train = math.ceil(cfg.preprocess.num_total_tokens / model.config.max_position_embeddings)
        assert len(train_dataset) > num_datasets_train, \
            f"num_total_tokens is {cfg.preprocess.num_total_tokens}. However, there are only {model.config.max_position_embeddings} * {len(train_dataset)} samples."
        print(f"num_total_tokens is {cfg.preprocess.num_total_tokens}.")
        print(f"num_datasets_train is {num_datasets_train}.")
        print(f"len(train_dataset) is {len(train_dataset)}.")
        train_dataset = train_dataset[:num_datasets_train]
        print(f"adjusted tokens: {model.config.max_position_embeddings} * {len(train_dataset)}")
        
    if rank == 0:
        logger.info("Preprocessing dataset has been done.")
        logger.info(f"The number of tokens in the dataset: {len(train_dataset)} * {model.config.max_position_embeddings}")

    sampler1 = DistributedSampler(train_dataset, rank=rank, num_replicas=world_size, shuffle=True)
    sampler2 = DistributedSampler(val_dataset, rank=rank, num_replicas=world_size)

    train_kwargs = {'batch_size': cfg.train.batch_size, 'sampler': sampler1}
    test_kwargs = {'batch_size': cfg.train.test_batch_size, 'sampler': sampler2}
    cuda_kwargs = {'num_workers': 2,
                    'pin_memory': True,
                    'shuffle': False}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

    train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
    val_loader = torch.utils.data.DataLoader(val_dataset, **test_kwargs)
 
    torch.cuda.set_device(local_rank)
    
    # Set up FSDP parameters
    mixed_precision_policy, t5_auto_wrap_policy = policies.get_policies(
        cfg.model.model_name,
        cfg.exec.mixed_precision, 
        cfg.exec.use_fp16, 
        rank,
        )

    if cfg.exec.sharding_strategy == "FULL_SHARD":
        sharding_strategy = ShardingStrategy.FULL_SHARD
    else:
        raise NotImplementedError
    
    if cfg.exec.checkpoint_type == "FULL_STATE_DICT":
        checkpoint_type = StateDictType.FULL_STATE_DICT
    else:
        raise NotImplementedError
    
    # Apply FSDP wrapping to the model
    model = FSDP(model,
        auto_wrap_policy=t5_auto_wrap_policy,
        mixed_precision=mixed_precision_policy,
        sharding_strategy=sharding_strategy,
        device_id=torch.cuda.current_device(),
        limit_all_gathers=cfg.exec.limit_all_gathers,
        cpu_offload=CPUOffload(offload_params=True),
        sync_module_states=True,
        param_init_fn=lambda module: module.to_empty(  # type: ignore
            device=torch.cuda.current_device(), recurse=False,  # type: ignore
        )
        if rank != 0
        else None,
        )
    
    if cfg.exec.fsdp_activation_checkpointing:
        policies.apply_fsdp_checkpointing(model, cfg.model.model_name)

    # Set up optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=cfg.train.lr)
    num_iters = len(train_loader) * cfg.train.epochs
    lr_warmup_steps = num_iters // 10
    logger.info(f"num_iters: {num_iters}, lr_warmup_steps: {lr_warmup_steps}")
    scheduler = WarmupCosineAnnealingLR(
        optimizer=optimizer,
        warmup_iterations=lr_warmup_steps,
        decay_iterations=num_iters,
        max_iterations=num_iters,
        eta_min=cfg.train.min_lr,
    )
    
    best_val_loss = float("inf")
    curr_val_loss = float("inf")

    for epoch in range(1, cfg.train.epochs + 1):
        train_loss = train(model, rank, train_loader, optimizer, scheduler, epoch, sampler=sampler1)
        if cfg.train.run_validation:
            curr_val_loss = validation(model, rank, val_loader)

        if rank == 0:
            print(f"--> epoch {epoch} completed...entering save and stats zone")
            wandb.log({"train_loss": train_loss.item()})

            if cfg.train.run_validation:
                wandb.log({"val_loss": curr_val_loss.item()})


        if cfg.train.save_model and curr_val_loss < best_val_loss:
            
            if checkpoint_type == StateDictType.FULL_STATE_DICT:
                model_checkpointing.save_model_checkpoint(
                    model, tokenizer, optimizer, rank, 
                    checkpoint_type, 
                    output_path,
                    epoch=epoch,
                )
            elif checkpoint_type == StateDictType.SHARDED_STATE_DICT:
                model_checkpointing.save_model_and_optimizer_sharded(
                    model, rank, 
                    output_path,
                    )
                if cfg.exec.save_optimizer:
                    model_checkpointing.save_model_and_optimizer_sharded(
                        model, rank, 
                        output_path,
                        optim=optimizer
                        )

            if cfg.exec.save_optimizer:
                model_checkpointing.save_optimizer_checkpoint(
                    model, optimizer, rank, 
                    output_path, 
                    epoch=epoch
                )     

        if curr_val_loss < best_val_loss:
            best_val_loss = curr_val_loss
            if rank==0:
                print(f"-->>>> New Val Loss Record: {best_val_loss}")

    if rank==0:
        run.finish()

    dist.barrier()
    clean_up()

if __name__ == "__main__":
    main()