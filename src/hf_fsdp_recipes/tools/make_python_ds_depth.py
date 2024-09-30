from datasets import load_dataset
from transformers import AutoTokenizer
import fire
import logging
import ast
import matplotlib.pyplot as plt
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

def calculate_ast_depth(node, depth=0):
    if not isinstance(node, ast.AST):
        return depth
    return max((calculate_ast_depth(child, depth+1) for child in ast.iter_child_nodes(node)), default=depth)

def calc_depth(content: str):
    parsed_code = ast.parse(content)
    # ASTの深さを計算
    depth = calculate_ast_depth(parsed_code)
    return depth

def check_desired_depth(example_batch, low:int=0, high:int=1000):
    l = -1
    bool_list = []
    for i in example_batch["content"]:
        try:
            l = calc_depth(i)
        except:
            pass

        if l == -1:
            bool_list.append(False)
        elif low < l and l < high:
            bool_list.append(True)
        else:
            bool_list.append(False)

    return bool_list

def main():
    raw_datasets = load_dataset(
        "bigcode/the-stack", 
        data_dir="data/python", 
        split="train",
        ignore_verifications=True
        )

    short_dataset = raw_datasets.filter(
        check_desired_depth, 
        batched=True, 
        num_proc=10,
        fn_kwargs={"low":0, "high":8}
        )
    short_dataset.save_to_disk("/home/uchiyama.fumiya/ucllm/ucllm/program/outputs/data/python-0-8")

    del short_dataset

    middle_dataset = raw_datasets.filter(
        check_desired_depth, 
        batched=True, 
        num_proc=10,
        fn_kwargs={"low":8, "high":12}
        )
    middle_dataset.save_to_disk("/home/uchiyama.fumiya/ucllm/ucllm/program/outputs/data/python-8-12")

    del middle_dataset

    long_dataset = raw_datasets.filter(
        check_desired_depth, 
        batched=True, 
        num_proc=10,
        fn_kwargs={"low":12, "high":20}
        )
    long_dataset.save_to_disk("/home/uchiyama.fumiya/ucllm/ucllm/program/outputs/data/python-12-20")


if __name__=='__main__':
    fire.Fire(main)
