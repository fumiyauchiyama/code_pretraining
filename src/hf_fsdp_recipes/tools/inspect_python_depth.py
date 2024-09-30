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

def main():
    raw_datasets = load_dataset(
        "bigcode/the-stack", 
        data_dir="data/python", 
        split="train",
        ignore_verifications=True
        )
    
    raw_datasets = raw_datasets.shuffle(seed=42)

    raw_datasets = raw_datasets.filter(lambda example: len(example["content"].strip()) > 0)
    
    raw_datasets = raw_datasets.select(range(50000))

    data = []
    failures = 0
    for i in tqdm(range(len(raw_datasets))):
        try:
            l = calc_depth(raw_datasets[i]['content'])
            data.append(l)
        except:
            print("failed")
            failures += 1

    print("Failures:", failures)

    data.sort()
    print(data[len(data)//3])
    print(data[len(data)*2//3])
    print(data[len(data)-1])

    count = 0
    for d in reversed(data):
        if d > 50:
            count += 1
        else:
            break
    print("Count more than 50: ",count)

    # plt.hist(data, bins=range(min(data), max(data) + 2), align='left', color='blue', edgecolor='black')
    plt.hist(data, bins=range(min(data), 50), align='left', color='blue', edgecolor='black')
    plt.xlabel('AST Depth')
    plt.ylabel('Frequency')
    plt.savefig('the_stack_frequency_histogram_50000.png')



if __name__=='__main__':
    fire.Fire(main)
