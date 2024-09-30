import os
import boto3
from smart_open import open
from datasets import load_dataset

session = boto3.Session()
s3 = session.client("s3")

def download_contents(files):
    for file in files:
        s3_url = f"s3://softwareheritage/content/{file['blob_id']}"
        with open(s3_url, "rb", compression=".gz", transport_params={"client": s3}) as fin:
            file["content"] = fin.read().decode(file["src_encoding"])
    
    return {"files": files}

ds = load_dataset("bigcode/the-stack-v2-train-full-ids", data_dir="data/Dockerfile", split="train")
ds = ds.map(lambda row: download_contents(row["files"]))
for row in ds:
    for file in row["files"]:
        print(file["content"])
    break
