# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from tqdm import tqdm
from itertools import chain

import torch

class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, chunk_size=4096, preprocess=True):
        self.chunk_size = chunk_size

        self.samples = []

        buffer = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
            }

        if preprocess:
            for sample in tqdm(dataset, desc="Preprocessing dataset", dynamic_ncols=True):
                buffer = {k: v + sample[k] for k,v in buffer.items()}

                while len(next(iter(buffer.values()))) > self.chunk_size:
                    self.samples.append({k: torch.tensor(v[:self.chunk_size]) for k,v in buffer.items()})
                    buffer = {k: v[self.chunk_size:] for k,v in buffer.items()}
        else:
            for sample in tqdm(dataset, desc="Loading dataset", dynamic_ncols=True):
                self.samples.append({k: torch.tensor(v) for k,v in sample.items()})
        

    def __getitem__(self, idx):
        return self.samples[idx]

    def __len__(self):
        return len(self.samples)
