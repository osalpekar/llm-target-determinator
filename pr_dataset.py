#!/usr/bin/env python3

import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from get_tokens_from_directory import get_tokens_from_file, should_process_file

from pr_tokenization import PTTokenizer

from torch.utils.data import Dataset


DIFF_REGEX = re.compile(r"^diff --git a/(?P<a>.+) b/(?P<b>.+)$")
LINENO_REGEX = re.compile(r"@@ -(?P<begin>\d+),\d+ \+(?P<end>\d+),\d+ @@")


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

        current_file = None
        selected_lines = []
        all_tokens = defaultdict(list)

        for line in patch.split("\n"):
            mf = DIFF_REGEX.match(line)
            if mf:
                if current_file:
                    tokens = get_tokens_from_file(
                        Path(current_file),
                        repo_dir=None,
                        tests_only=False,
                        selected_lines=selected_lines,
                    )
                    all_tokens.update(tokens)

                # Reset for the next file
                current_file = None
                selected_lines = []

                filepath = mf["a"]
                current_file = f"{self.dir}/{pr_number}/{filepath}"
                if not should_process_file(
                    str(os.path.basename(current_file)),
                    str(os.path.dirname(current_file)),
                    "",
                ):
                    # Not interested in this file, i.e. cpp
                    current_file = None
                continue

            ml = LINENO_REGEX.match(line)
            if not ml:
                continue

            begin_lineno = int(ml["begin"])
            end_lineno = int(ml["end"])
            selected_lines.append((begin_lineno, end_lineno))

        if current_file:
            tokens = get_tokens_from_file(
                Path(current_file),
                repo_dir=None,
                tests_only=False,
                selected_lines=selected_lines,
            )
            all_tokens.update(tokens)

        return all_tokens


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
        print(r)


if __name__ == "__main__":
    main()
