import ast
import json
import math
import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, NamedTuple, Tuple

import torch
from preproc import get_functions
from tokenizer import Tokenizer
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parent

# The size of inputs to the model for inference is variable. Naively, it's
# shape is (num_functions_in_file x context_length). Different files have
# different numbers of functions, and a number of files have hundreds or
# thousands of functions. Memory utilization (size of intermediate activations,
# size of batch, etc.) is therefore correlated with the input size. Since we
# may run this model on lower-memory GPUs such as 24GB A10G's, we ensure we can
# still complete indexing by partitioning files up-front and then indexing the
# partitioned files, which has lower memory utilization.


class FilePartition(NamedTuple):
    """
    Stores metadata about a partition of a file.
    partition_id is 1-indexed, must be 0 < partition_id <= num_partitions

    Un-partitioned files have partition_id = num_partitions = 1
    """

    filename: str
    partition_id: int
    num_partitions: int


class UnittestDataset(Dataset):
    def __init__(self, config):
        self.config = config
        self.filelist = self.create_filelist()
        self.tokenizer = Tokenizer(self.config)

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
        pass

    def tokenize_items(self, items: List[Tuple[str, str]]):
        token_list = []
        names = []
        for name, item in items:
            tokens = self.tokenizer.encode(item)
            token_list.append(tokens)
            names.append(name)

        # Concatenate the tokenized vectors from each item into a single
        # matrix of shape (num_items x embedding_dimension)
        if self.config.model == "codellama":
            num_items = len(token_list)

            # Create a tensor of shape (num_items x CONTEXT_LENGTH) filled
            # with pad_id
            tensor = torch.full(
                (num_items, self.config.max_context_len),
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

            return {"tokens": tensor, "attn_mask": attn_mask}, names
        else:
            return torch.cat(token_list), names


class FunctionGranularityDataset(UnittestDataset):
    """
    Dataset where a function from a file is a signular item.  Each function gets
    its own tokenization and embedding.
    """

    def __init__(self, config):
        super().__init__(config)
        self.file_partitions: List[FilePartition] = flatten(
            [self.partition_file(file) for file in self.filelist]
        )

        # Store results from the ast parser in an attempt to reduce number of
        # times a file is parsed
        self.functions_cache = {}

    def partition_file(self, filename) -> List[FilePartition]:
        threshold = 100
        # Approximate number of functions in file by counting number of
        # functions defs.  Partition file based on this number.  Each partition
        # should have <100 functions.
        partitions = []
        with open(filename) as f:
            text = f.read()
        occurrences = text.count("def ")
        if occurrences and occurrences > threshold:
            num_partitions = math.ceil(occurrences / threshold)
            for i in range(num_partitions):
                partitions.append(
                    FilePartition(filename, i + 1, num_partitions)
                )
        else:
            partitions.append(FilePartition(filename, 1, 1))
        return partitions

    def __len__(self):
        return len(self.file_partitions)

    def __getitem__(self, idx):
        filename, partition, num_partitions = self.file_partitions[idx]

        # Use function info from a previous parsing of the ast in cache if
        # possible
        if filename in self.functions_cache:
            functions_in_file = self.functions_cache[filename]
        else:
            functions_in_file = sorted(get_functions(filename).items())
            self.functions_cache[filename] = functions_in_file

        num_functions = len(functions_in_file)
        size = math.ceil(num_functions / num_partitions)
        start_i = size * (partition - 1)
        end_i = start_i + size
        functions = functions_in_file[start_i:end_i]

        # Some test files don't actually have any unittest functions. We handle
        # that case here.
        if len(functions) == 0:
            empty_tensor = torch.tensor([], dtype=torch.int64).reshape(
                0, self.config.max_context_len
            )
            return ({"tokens": empty_tensor, "attn_mask": empty_tensor}, [])

        # Get tokens for each function
        return self.tokenize_items(functions)


class FileGranularityDataset(UnittestDataset):
    """
    Dataset where the entire file is the item.  An entire file gets tokenized
    and embedded together.
    """

    def __init__(self, config):
        super().__init__(config)

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        filename = self.filelist[idx]
        with open(filename) as f:
            text = f.read()

        return self.tokenize_items([(filename, text)])


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
