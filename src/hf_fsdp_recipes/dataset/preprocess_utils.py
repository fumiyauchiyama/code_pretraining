from datasets import Dataset
import logging
from typing import Optional
import torch

logger = logging.getLogger()

def limit_sample_num(dataset: Dataset, num_total_samples:int) -> Dataset:
    assert num_total_samples <= len(dataset), \
            f"num_total_samples is {num_total_samples}. However, there are only {len(dataset)} samples."
    
    return dataset.select(range(num_total_samples))

def avoid_short_samples(
        dataset: Dataset,
        num_total_samples: int,
        min_token_length: int,
        content_column_name: str,
        tokenizer,
        is_shuffled: bool=False,
        seed: int=0
        ) -> Dataset:
    # Since filtering by token consumes lots of time, 
    # limits the sample number into 10 * num_total_samples.
    if len(dataset) >= 10 * num_total_samples:
        logger.info(f"total samples are too large to filter: {len(dataset)}.")

        if not is_shuffled:
            logger.info("Shuffling to select random samples from whole dataset")

            dataset = dataset.shuffle(
                seed=seed
                )
        
        logger.info(f"limits sample number to 10 * {num_total_samples}...")
        dataset = dataset.select(
            range(10 * num_total_samples)
            )
        
    logger.info(f"total samples before filter: {len(dataset)}")

    dataset = dataset.filter(
        lambda example: \
            len(tokenizer.encode(example[content_column_name])) \
                >= min_token_length
        , num_proc=10)
    
    logger.info(f"total samples after filter: {len(dataset)}")
    
    return dataset

def insert_special_token(
        dataset: Dataset,
        content_column_name: str,
        bos_token: Optional[str]=None,
        eos_token: Optional[str]=None,
        ):
    if bos_token is not None:
        dataset = dataset.map(
            lambda sample: \
            {content_column_name: bos_token + sample[content_column_name]}
        )

    if eos_token is not None:
        dataset = dataset.map(
            lambda sample:
            {content_column_name: sample[content_column_name] + eos_token}
        )
    
    return dataset

def tokenize_with_padding(
        dataset: Dataset,
        tokenizer,
        content_column_name: str,
        max_length: int
        ) -> dict[str, torch.tensor]:
    """return tokenized dataset in the form of torch.Tensor"""

    if tokenizer.pad_token is None:
        logger.info("set pad_token=eos_token since tokenizer has no pad_token")
        tokenizer.pad_token = tokenizer.eos_token
    
    def tokenize(element):
        # modelconfigのmax_position_embeddingsにする
        outputs = tokenizer(
            element[content_column_name],
            padding=True, 
            truncation=True,
            max_length=max_length,
            return_overflowing_tokens=True,
            return_tensors='pt',
            add_special_tokens=False
        )
        return outputs
    
    tokenized_dataset =  dataset.map(
        tokenize, 
        batched=True, 
        batch_size=2000,
        num_proc=10,
        remove_columns=dataset.column_names
    )
    print(tokenized_dataset[0])

    return tokenized_dataset


def tokenize(
        dataset: Dataset,
        tokenizer,
        content_column_name: str,
        ) -> dict[str, list[int]]:
    """Simple tokenize function returns """
    
    def simple_tokenize(element):
        outputs = tokenizer(
            element[content_column_name],
            add_special_tokens=False
        )
        return outputs
    
    return dataset.map(
        simple_tokenize, 
        batched=True, 
        batch_size=2000,
        num_proc=10,
        remove_columns=dataset.column_names
    )
