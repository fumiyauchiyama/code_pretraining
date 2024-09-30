from datasets import load_from_disk, Dataset
from transformers import AutoTokenizer

import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument("--path", type=str)
    parser.add_argument("--name", type=str, default=None)

    args = parser.parse_args()

    d_train = load_from_disk(args.path + "/train")
    d_train.push_to_hub(args.name,split="train", private=True)

    del d_train

    d_val = load_from_disk(args.path + "/val")
    d_val.push_to_hub(args.name,split="validation", private=True)

if __name__ == "__main__":
    main()