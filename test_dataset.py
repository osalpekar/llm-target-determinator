import ast
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path
import math

import torch
from preproc import get_functions
from tokenizer import Tokenizer

from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parent

from typing import Dict, Tuple, List
from torch.utils.data import DataLoader, Dataset


class UnittestDataset(Dataset):
    def __init__(self, config):
        self.config = config

        self.filelist = self.create_filelist()
        self.tokenizer = Tokenizer(self.config)

        self.partitions = flatten(
            [self.partition_file(file) for file in self.filelist]
        )
        # Store results from the ast parser in an attempt to reduce number of
        # times a file is parsed
        self.functions = {}

    def create_filelist(self):
        """
        Returns the list of files of interest in the specified subdirectory.
        We sort the generated list so that each rank deterministically ends up
        with the same filelist, which ensures the Dataloder and
        DistributedSampler correctly cover all the relevant files.
        """
        all_files = []
        project_dir = Path(self.config.project_dir).expanduser()

        for root, dirs, files in os.walk(self.config.project_dir):
            for file in files:
                if self.should_include_file(file, root):
                    file_path = os.path.join(root, file)
                    all_files.append(file_path)

        return sorted(all_files)

    def should_include_file(self, file, root):
        """
        Returns whether the specified file should be included in the file list.
        Note that we are only interested in Python files with the specified prefix,
        provided those files do not lie in certain directories like third_party.
        """
        return (
            file.endswith(".py")
            and file.startswith(self.config.file_prefix)
            and "third_party" not in root
        )

    def __len__(self):
        return len(self.partitions)

    def tokenize_functions(self, functions):
        token_list = []
        for signature in functions:
            function_body = functions[signature]
            tokens = self.tokenizer.encode(function_body)
            token_list.append(tokens)

        # Concatenate the tokenized vectors from each function into a single
        # matrix of shape (num_functions x embedding_dimension)
        if self.config.model == "codellama":
            num_functions = len(token_list)

            # Create a tensor of shape (num_functions x CONTEXT_LENGTH) filled
            # with pad_id
            tensor = torch.full(
                (num_functions, self.config.max_context_len),
                self.tokenizer.pad_id,
                dtype=torch.long,
            )
            print(f"pad: {self.tokenizer.pad_id}")

            for k, t in enumerate(token_list):
                # truncate token list to max context length
                t = t[: self.config.max_context_len]
                # insert the tokens into the empty tensor
                tensor[k, : len(t)] = torch.tensor(t, dtype=torch.long)

            attn_mask = torch.where(tensor == self.tokenizer.pad_id, 0.0, 1.0)

            return {"tokens": tensor, "attn_mask": attn_mask}
        else:
            return torch.cat(token_list)

    def partition_file(self, filename):
        threshold = 100
        # Is this stupid? yes
        # Approximate number of functions in file by counting number of
        # functions defs.  Partition file based on this number.  Each partition
        # should have <100 functions.
        partitions = []
        with open(filename) as f:
            text = f.read()
        occurances = text.count("def ")
        if occurances and occurances > threshold:
            num_partitions = math.ceil(occurances / threshold)
            for i in range(num_partitions):
                partitions.append((filename, i + 1, num_partitions))
        else:
            partitions.append((filename, 1, 1))
        return partitions

    def __getitem__(self, idx):
        filename, partition, num_partitions = self.partitions[idx]

        if filename in self.functions:
            functions_in_file = self.functions[filename]
        else:
            functions_in_file = sorted(get_functions(filename).items())
            self.functions[filename] = functions_in_file

        num_functions = len(functions_in_file)
        size = math.ceil(num_functions / num_partitions)
        start_i = size * (partition - 1)
        end_i = start_i + size
        functions = functions_in_file[start_i:end_i]
        functions = {x: y for x, y in functions}

        # Some test files don't actually have any unittest functions. We handle
        # that case here.
        if len(functions) == 0:
            empty_tensor = torch.tensor([], dtype=torch.int64).reshape(
                0, self.config.max_context_len
            )
            return ({"tokens": empty_tensor, "attn_mask": empty_tensor}, [])

        # Get tokens for each function
        tokens = self.tokenize_functions(functions)
        function_list = list(functions.keys())

        return tokens, function_list


def flatten(lst):
    return [item for sublist in lst for item in sublist]


def collate_fn(data):
    examples_tokens = [item[0]["tokens"] for item in data]
    examples_masks = [item[0]["attn_mask"] for item in data]
    funclist = flatten([item[1] for item in data])
    examples = {
        "tokens": torch.cat(examples_tokens),
        "attn_mask": torch.cat(examples_masks),
    }
    return examples, funclist


# Small Test
# dataset = UnittestDataset("assets/filelist.json")
# dataloader = DataLoader(dataset, collate_fn=collate_fn, batch_size=2)

# for idx, batch in enumerate(dataloader, 0):
#     data, funcs = batch
#     print(data.shape)
#     print(len(funcs))
#     if idx == 0:
#         sys.exit(0)
