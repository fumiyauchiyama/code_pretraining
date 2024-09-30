import fire
from datasets import load_dataset, Dataset
import matplotlib.pyplot  as plt
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer
)
import math
from tqdm import tqdm

def count_token_kinds(d: Dataset, tokenizer: PreTrainedTokenizer)->list[int]:
    token_list = [0 for _ in range(len(tokenizer))]
    for sample in tqdm(d):
        for token in sample['input_ids']:
            token_list[token] += 1
    return token_list

def top_n_tokens(
        token_list: list[int], 
        tokenizer: PreTrainedTokenizer, 
        n:int = 15
        ) -> list[str]:
    token_ranking = sorted(
        range(len(token_list)), 
        key=token_list.__getitem__, 
        reverse=True
        )[:n]
    return tokenizer.convert_ids_to_tokens(token_ranking)
    
def draw_plot(data: list[int], save_name: str) -> None:
    plt.plot(data)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram')
    plt.grid(True)
    plt.savefig(f"/share5/uchiyama.fumiya/ucllm/ucllm/program/logs/datasets/{save_name}_plot.png")

def draw_hist(data: list[int], save_name: str) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.hist(data, bins=bin)
    ax.set_title('first histogram')
    ax.set_xlabel('x')
    ax.set_ylabel('freq')
    fig.savefig(f"/share5/uchiyama.fumiya/ucllm/ucllm/program/logs/datasets/{save_name}_hist.png")


def main(
        dataset_name:str,
        dataset_subname:str = "",
        thestack_language:str = "",
        num_total_samples:str = 50000,
        min_token_length:int = 0,
        seed:int = 42,
        base_model:str = "openai-community/gpt2",
        bin:int = 10,
        max_length: int = 10000,
        context_length:int = 1024
        ):
    if dataset_name == "wikitext":
        raw_datasets = load_dataset(
            dataset_name, 
            dataset_subname, 
            split="train"
            )
        content_column_name = "text"
        save_name = f"gpt2_wikitext_{dataset_subname}"
    elif dataset_name == "allenai/c4":
        raw_datasets = load_dataset(
            dataset_name, 
            dataset_subname, 
            split="train"
            )
        content_column_name = "text"
        save_name = f"gpt2_c4_{dataset_subname}"
    elif dataset_name == "bigcode/the-stack":
        assert thestack_language is not None
        raw_datasets = load_dataset(
            "bigcode/the-stack", 
            data_dir=f"data/{thestack_language}", 
            split="train"
            )
        content_column_name = "content"
        save_name = f"gpt2_thestack_{thestack_language}"
    else:
        raise NotImplementedError
    
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

    raw_datasets = raw_datasets.filter(lambda example: len(example[content_column_name]) > min_token_length)

    assert num_total_samples <= len(raw_datasets), f"num_total_samples is {num_total_samples}. However, there are only {len(raw_datasets)} samples for {dataset_name}."

    raw_datasets = raw_datasets.shuffle(
        seed=seed
        ).select(
            range(num_total_samples)
            )
    
    raw_datasets = raw_datasets.train_test_split(
        test_size=0.01, 
        seed=seed
        )
    
    def tokenize(element):
        outputs = tokenizer(
            element[content_column_name],
            truncation=True,
            max_length=context_length,
            return_overflowing_tokens=True,
            return_length=True,
        )
        return outputs
    
    tokenized_datasets = raw_datasets.map(
        tokenize, 
        batched=True, 
        batch_size=2000,
        num_proc=10,
        remove_columns=raw_datasets["train"].column_names
    )
    
    print(tokenized_datasets["train"]["length"])

    token_types = count_token_kinds(tokenized_datasets["train"], tokenizer)
    draw_plot(token_types, save_name)

    print(top_n_tokens(token_types, tokenizer))

    

if __name__ == '__main__':
    fire.Fire(main)