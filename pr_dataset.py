#!/usr/bin/env python3

import json
from typing import Any

from get_tokens_from_directory import get_tokens_from_directory
from pr_tokenization import PTTokenizer

from torch.utils.data import Dataset


class PyTorchPRDataset(Dataset):
    def __init__(self, pull_requests_file: str, pull_requests_dir: str):
        self.tokenizer = PTTokenizer()

        with open(pull_requests_file) as f:
            self.pull_requests = json.load(f)
        self.dir = pull_requests_dir

    def __len__(self):
        return len(self.pull_requests)

    def __getitem__(self, idx):
        patch = self.pull_requests[idx]["patch"]
        pr_number = self.pull_requests[idx]["number"]
        tokenized_functions = get_tokens_from_directory(
            f"{self.dir}/{pr_number}",
            repo_dir=None,
            file_prefix="",
            tests_only=False,
        )
        return tokenized_functions


def parse_args() -> Any:
    from argparse import ArgumentParser

    parser = ArgumentParser("GitHub PR tokenization")
    parser.add_argument("--input", type=str, help="the input JSON file")
    parser.add_argument("--pr-dir", type=str, help="the PR data directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = PyTorchPRDataset(args.input, args.pr_dir)
    for r in data:
        print(len(r))


if __name__ == "__main__":
    main()
