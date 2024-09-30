from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer

import fire
import torch
import os
import sys
import time
import logging
import wandb
import math
from typing import Optional
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

def main(
    model_name: str, 
    path: str,
    name: Optional[str] = None,
    data_dir: Optional[str] = None,
    split: Optional[str] = None,
    verification_mode: Optional[str] = None,
    content_column_name: str = "text",
    preprocess: list[str] = ["avoid_short_sample", "shuffle", "prepend_eos", "train_test_split", "pack_texts"],
    seed: int = 42,
    num_total_samples: int = 1e18,
    ctx_length: int = 2048,
    num_total_tokens: int = 100_000_000_000,
    output_dir: str = "outputs/processed_data"
    ) -> None:

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    raw_datasets = load_dataset(path, name=name, data_dir=data_dir, split=split, verification_mode=verification_mode)

    assert content_column_name in raw_datasets.column_names

    output_dir += "/" + path
    if name is not None:
        output_dir += "/" + name
    if data_dir is not None:
        output_dir += "/" + data_dir
    
    if len(preprocess) > 0:
        for idx, i in enumerate(preprocess):

            logger.info(f"preprocess: {i}")

            if i == "avoid_short_sample":
                raw_datasets = raw_datasets.filter(lambda example: len(example[content_column_name]) > 0)

            elif i == "shuffle":
                raw_datasets = raw_datasets.shuffle(
                    seed=seed
                    )
            elif i == "select":
                if num_total_samples <= len(raw_datasets):
                    f"num_total_samples is {num_total_samples}. However, there are only {len(raw_datasets)} samples. Skipping select process"
                else:
                    raw_datasets = raw_datasets.select(
                        range(select.num_total_samples)
                        )
            elif i == "prepend_eos":
                def prepend_eos(sample):
                    contents = []
                    for example in sample[content_column_name]:
                        contents.append(tokenizer.eos_token + example)

                    return {content_column_name: contents}

                raw_datasets = raw_datasets.map(
                    prepend_eos,
                    batched=True, 
                    batch_size=2000,
                    num_proc=10,
                    remove_columns=raw_datasets.column_names
                    )
            elif i == "train_test_split":
                raw_datasets = raw_datasets.train_test_split(
                    test_size=0.01,
                    seed=42,
                    )
            else :
                pass


    assert "train" in raw_datasets.keys() \
        and "test" in raw_datasets.keys()
    
    train_dataset = raw_datasets["train"]
    val_dataset = raw_datasets["test"]

    if "pack_texts" in preprocess:
        add_label = lambda sample: \
            {"labels": sample["input_ids"].copy()}
        # ConcatDataset converts to torch.tensor
        train_dataset = tokenize(
            train_dataset, 
            tokenizer,
            content_column_name,
            )
        train_dataset = train_dataset.map(
            add_label,
            batched=True, 
            batch_size=2000,
            num_proc=10,
            )
        train_dataset = ConcatDataset(
            train_dataset,
            ctx_length
            )
        val_dataset = tokenize(
            val_dataset, 
            tokenizer,
            content_column_name,
            )
        val_dataset = val_dataset.map(
            add_label,
            batched=True, 
            batch_size=2000,
            num_proc=10,
            )
        val_dataset = ConcatDataset(
            val_dataset,
            ctx_length,
            )

        if num_total_tokens is not None:
            num_datasets_train = math.ceil(num_total_tokens / ctx_length)
            if len(train_dataset) > num_datasets_train:
                logger.info(f"num_total_tokens is {num_total_tokens}. However, there are only {ctx_length} * {len(train_dataset)} samples.")
            else:
                train_dataset = train_dataset[:num_datasets_train]

        train_dataset.save(output_dir + '/train.pth')
        val_dataset.save(output_dir + '/val.pth')

    else:
        add_label = lambda sample: \
            {"labels": sample["input_ids"].copy()}
        
        train_dataset = tokenize_with_padding(
            train_dataset,
            tokenizer,
            content_column_name,
            ctx_length,
            )
        print(train_dataset[0])
        train_dataset = train_dataset.map(
            add_label,
            batched=True, 
            batch_size=2000,
            num_proc=10,
            )
        val_dataset = tokenize_with_padding(
            val_dataset,
            tokenizer,
            content_column_name,
            ctx_length,
            )
        val_dataset = val_dataset.map(
            add_label,
            batched=True, 
            batch_size=2000,
            num_proc=10,
            )

        train_dataset.save_to_disk(output_dir + "/train", max_shard_size="2GB")
        val_dataset.save_to_disk(output_dir  + "/val", max_shard_size="2GB")
        



if __name__ == "__main__":
    fire.Fire(main)