import ast
import json
import os
import sys

import torch
from pr_tokenization import PTTokenizer, CONTEXT_LENGTH

from torch.utils.data import DataLoader, Dataset


class TestDataset(Dataset):
    def __init__(self, filelist):
        with open(filelist) as f:
            filelist_json = json.load(f)

        self.filelist = filelist_json["all_files"]
        self.tokenizer = PTTokenizer("bert-base-uncased")

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        filename = self.filelist[idx]
        print(filename)
        with open(filename) as f:
            content = f.read()

        module = ast.parse(content)

        functions = {}

        # It takes about 36 minutes to do the ast function parsing for the
        # entire pytorch directory. This is likely prohibitive, so best if we
        # do it in dataloader.
        # By the end of this for-loop, we have a map of all functions in the
        # file and the function bodies
        for node in ast.walk(module):
            if isinstance(node, ast.FunctionDef):
                # If the node is a function, extract its name and its arguments
                signature = node.name
                body = ast.get_source_segment(content, node)
                functions[signature] = body
            elif isinstance(node, ast.ClassDef):
                # If the node is a class, we also want to get its methods
                for sub_node in node.body:
                    if isinstance(sub_node, ast.FunctionDef):
                        signature = node.name + "." + sub_node.name
                        body = ast.get_source_segment(content, sub_node)
                        functions[signature] = body

        if len(functions) == 0:
            return torch.tensor([], dtype=torch.int64).reshape(0, CONTEXT_LENGTH)

        # Get tokens for each function
        token_list = []
        for signature in functions:
            function_body = functions[signature]
            tokens = self.tokenizer.encode(function_body)
            token_list.append(tokens)

        # Concatenate the tokenized vectors from each function into a single
        # matrix of shape (num_functions x embedding_dimension)
        return torch.cat(token_list)


def collate_fn(data):
    return torch.cat(data)


# dataset = TestDataset("assets/filelist.json")
# dataloader = DataLoader(dataset, collate_fn=collate_fn, batch_size=2)


# Small Test
# for idx, batch in enumerate(dataloader, 0):
#     if idx == 0:
#         sys.exit(0)
