#!/usr/bin/env python3

import json
from transformers import AutoTokenizer
from torch.utils.data import Dataset
from typing import Any


class PyTorchPRDataset(Dataset):
    def __init__(self, pull_requests_file: str):
        self.checkpoint = "bigcode/starcoderplus"
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)

        with open(pull_requests_file) as f:
            self.pull_requests = json.load(f)

    def __len__(self):
        return len(self.pull_requests)

    def __getitem__(self, idx):
        # TODO: Decide on what the input here would look like. This uses the following dummy JSON
        # pytorch_prs_with_patch_100.json
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
