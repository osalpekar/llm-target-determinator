#!/usr/bin/env python3

import json
from typing import Any

from torch.utils.data import Dataset
from transformers import AutoTokenizer

from torch.utils.data import Dataset
from transformers import AutoTokenizer

class PTTokenizer:
    def __init__(self, model_checkpoint: str = "bigcode/starcoderplus"):
        self.model_checkpoint = model_checkpoint
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint, use_auth_token=True)
        self.tokenizer.pad_token = "[PAD]"

    def encode(self, data: str) -> Any:
        return self.tokenizer.encode(data, return_tensors="pt", padding=True)

    def decode(self, tokenized_data) -> str:
        return self.tokenizer.decode(tokenized_data)

class PyTorchPRDataset(Dataset):
    def __init__(self, pull_requests_file: str):
        self.tokenizer = PTTokenizer()

        with open(pull_requests_file) as f:
            self.pull_requests = json.load(f)

    def __len__(self):
        return len(self.pull_requests)

    def __getitem__(self, idx):
        # TODO: Decide on what the input here would look like. This uses the
        # following dummy JSON pytorch_prs_with_patch_100.json
        patch = self.pull_requests[idx]["patch"]
        return self.tokenizer.encode(patch)


def parse_args() -> Any:
    from argparse import ArgumentParser

    parser = ArgumentParser("GitHub PR tokenization")
    parser.add_argument("--input", type=str, help="the input JSON file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = PyTorchPRDataset(args.input)
    for r in data:
        print(r)


if __name__ == "__main__":
    main()
