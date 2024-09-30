from datasets import load_dataset
from transformers import AutoTokenizer
import fire
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

def main(dataset_name:str=None, base_model:str = "meta-llama/Llama-2-7b-hf"):
    if dataset_name == "c4":
        raw_datasets = load_dataset("allenai/c4", "en", split="train")
    elif dataset_name == "wikipedia":
        raw_datasets = load_dataset("wikipedia", "20220301.en", split="train")
    elif dataset_name == "fineweb":
        raw_datasets  = load_dataset("HuggingFaceFW/fineweb", name="CC-MAIN-2024-10", split="train")
    elif dataset_name == "slimpajama":
        raw_datasets = load_dataset(
            "cerebras/SlimPajama-627B"
        )
    elif dataset_name == "the-stack":
        raw_datasets = load_dataset(
            "bigcode/the-stack", 
            data_dir="data/ocaml", 
            split="train",
            ignore_verifications=True
            )
    elif dataset_name == "the-stack-v2":
        raw_datasets = load_dataset("bigcode/the-stack-v2-train-full-ids", split="train")
        return

    raw_datasets = raw_datasets.filter(lambda example: len(example["text"].strip()) > 2000)

    raw_datasets = raw_datasets.shuffle(seed=42).select(range(1000))
    
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        logger.info("Init pad_token since pad_token is None.")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

    def tokenize(element):
        return tokenizer(element["text"], return_overflowing_tokens=True, return_length=True)

    tokenized_datasets = raw_datasets.map(tokenize, batched=True, batch_size=2000, num_proc=10)

    total_length = sum([len(result['input_ids']) for result in tokenized_datasets])
    average_length = total_length / len(tokenized_datasets)

    logger.info(f"Average sequence length: {average_length}")

    print(raw_datasets)

    for sample in raw_datasets.select(range(10))['text']:
        logger.info(f"\n{sample}")
        print("-" * 20)

if __name__=='__main__':
    fire.Fire(main)
