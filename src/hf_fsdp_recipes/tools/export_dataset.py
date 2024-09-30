from datasets import load_dataset, Dataset
from transformers import AutoTokenizer

import torch
import os
import sys
import time
import logging
import wandb
import math

import argparse

current_path: str = os.getcwd()
sys.path.append(f"{current_path}/src")
sys.path.append(current_path)


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

def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset-path", type=str)
    parser.add_argument("--dataset-name", type=str, default=None)
    parser.add_argument("--dataset-data-dir", type=str, default=None)
    parser.add_argument("--verification-mode", type=str, default=None)
    parser.add_argument("--ctx-length", type=int, default=2048)
    parser.add_argument("--sample-num", type=int, default=None)
    parser.add_argument("--content-column-name", type=str, default="text")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tokenizer-name", type=str, default="openai-community/gpt2-xl")
    parser.add_argument("--train-test-split", type=float, default=0.01)
    parser.add_argument("--pack-texts", action="store_true")
    parser.add_argument("--num-total-tokens", type=int, default=100_000_000_000)
    parser.add_argument("--output-dir", type=str, default="outputs/data/")

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    output_name = ""

    if args.dataset_name is not None:
        output_name = args.dataset_name

    if args.dataset_data_dir is not None:
        output_name += args.dataset_data_dir

    output_name += args.tokenizer_name
    
    # Load dataset
    raw_datasets = load_dataset(
        args.dataset_path, name=args.dataset_name,
        data_dir=args.dataset_data_dir,
        verification_mode=args.verification_mode,
        token=True, split="train"
        )

    raw_datasets = raw_datasets.select_columns([args.content_column_name])

    raw_datasets = raw_datasets.filter(lambda example: len(example[args.content_column_name]) > 0)
    raw_datasets = raw_datasets.shuffle(
        seed=args.seed
        )

    # raw_datasets.save_to_disk(
    #     args.output_dir + output_name + "/filter",
    #     max_shard_size="1GB",
    #     num_proc=20,
    #     )

    if args.sample_num is not None:
        assert args.sample_num <= len(raw_datasets), \
            f"num_total_samples is {args.sample_num}. However, there are only {len(raw_datasets)} samples."
        
        raw_datasets = raw_datasets.select(
            range(args.sample_num)
            )
        output_name += str(args.sample_num)
            
    def prepend_eos(sample):
        sample[args.content_column_name] = tokenizer.eos_token + sample[args.content_column_name]
        return sample

    raw_datasets = raw_datasets.map(
        prepend_eos,
        num_proc=10,
        )
    
    # raw_datasets.save_to_disk(
    #     args.output_dir + output_name + "/filter-eos",
    #     max_shard_size="1GB",
    #     num_proc=20,
    #     )
    
    raw_datasets = raw_datasets.train_test_split(
        args.train_test_split
        )

    # raw_datasets.save_to_disk(
    #     args.output_dir + output_name + "/filter-eos-split",
    #     max_shard_size="1GB",
    #     num_proc=20,
    #     )
    
    assert "train" in raw_datasets.keys() \
        and "test" in raw_datasets.keys()
    
    train_dataset = raw_datasets["train"]
    val_dataset = raw_datasets["test"]

    del raw_datasets

    if args.pack_texts:
        add_label = lambda sample: \
            {"labels": sample["input_ids"].copy()}
        # ConcatDataset converts to torch.tensor
        print("tokenize")
        train_dataset = tokenize(
            train_dataset, 
            tokenizer,
            args.content_column_name,
            )
        print("add_label")
        train_dataset = train_dataset.map(
            add_label,
            batched=True, 
            batch_size=2000,
            num_proc=10,
            )
        print("ConcatDataset")
        train_dataset = ConcatDataset(
            train_dataset,
            args.ctx_length
            )
        print("tokenize")
        val_dataset = tokenize(
            val_dataset, 
            tokenizer,
            args.content_column_name,
            )
        print("val_dataset")
        val_dataset = val_dataset.map(
            add_label,
            batched=True, 
            batch_size=2000,
            num_proc=10,
            )
        print("ConcatDataset")
        val_dataset = ConcatDataset(
            val_dataset,
            args.ctx_length,
            )
    else:
        add_label = lambda sample: \
            {"labels": sample["input_ids"].copy()}
        
        train_dataset = tokenize_with_padding(
            train_dataset,
            tokenizer,
            args.content_column_name,
            args.ctx_length,
            )
        print(train_dataset[0])
        train_dataset = train_dataset.map(
            add_label,
            num_proc=10,
            )
        val_dataset = tokenize_with_padding(
            val_dataset,
            tokenizer,
            args.content_column_name,
            args.ctx_length,
            )
        val_dataset = val_dataset.map(
            add_label,
            num_proc=10,
            )

    if args.num_total_tokens is not None:
        num_datasets_train = math.ceil(args.num_total_tokens / args.ctx_length)
        if len(train_dataset) > num_datasets_train:
            train_dataset = train_dataset[:num_datasets_train]
        else:
            f"num_total_tokens is {args.num_total_tokens}. However, there are only {args.ctx_length} * {len(train_dataset)} samples."
        
    logger.info("Preprocessing dataset has been done.")
    logger.info(f"The number of tokens in the dataset: {len(train_dataset)} * {args.ctx_length}")


    if isinstance(train_dataset, ConcatDataset):

        train_dataset_hf = Dataset.from_list(train_dataset)
        val_dataset_hf = Dataset.from_list(val_dataset)

        train_dataset_hf.save_to_disk(
            args.output_dir + output_name + "/train",
            max_shard_size="1GB",
            )
        val_dataset_hf.save_to_disk(
            args.output_dir + output_name + "/val",
            max_shard_size="1GB",
            )


    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()