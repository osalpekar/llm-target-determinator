import ast
import json
import os
import sys
from collections import OrderedDict

import torch
from pr_tokenization import CONTEXT_LENGTH, PTTokenizer

from torch.utils.data import DataLoader, Dataset

# TODO: Test -> Unittest
class UnittestDataset(Dataset):
    def __init__(self, filelist):
        with open(filelist) as f:
            filelist_json = json.load(f)

        self.filelist = filelist_json["all_files"]
        self.tokenizer = PTTokenizer("bert-base-uncased")

    def __len__(self):
        return len(self.filelist)

    def get_functions(self, filename, file_content):
        functions = OrderedDict()

        module = ast.parse(file_content)

        # It takes about 36 minutes to do the ast function parsing for the
        # entire pytorch directory. This is likely prohibitive, so best if we
        # do it in dataloader.
        # By the end of this for-loop, we have a map of all functions in the
        # file and the function bodies
        for node in ast.walk(module):
            if isinstance(node, ast.FunctionDef):
                # If the node is a function, extract its name and its arguments
                signature = filename + ":" + node.name
                body = ast.get_source_segment(file_content, node)
                functions[signature] = body
            elif isinstance(node, ast.ClassDef):
                # If the node is a class, we also want to get its methods
                for sub_node in node.body:
                    if isinstance(sub_node, ast.FunctionDef):
                        signature = filename + ":" + node.name + "." + sub_node.name
                        body = ast.get_source_segment(file_content, sub_node)
                        functions[signature] = body

        return functions

    def tokenize_functions(self, functions):
        token_list = []
        for signature in functions:
            function_body = functions[signature]
            tokens = self.tokenizer.encode(function_body)
            token_list.append(tokens)

        # Concatenate the tokenized vectors from each function into a single
        # matrix of shape (num_functions x embedding_dimension)
        return torch.cat(token_list)

    def __getitem__(self, idx):
        filename = self.filelist[idx]
        print(filename)

        with open(filename) as f:
            # TODO: filelist can just have relative paths and we can call
            # expanduser here just before opening the file
            file_content = f.read()

        # Get functions from the file
        functions = self.get_functions(filename, file_content)

        # Some test files don't actually have any unittest functions. We handle
        # that case here.
        if len(functions) == 0:
            return (torch.tensor([], dtype=torch.int64).reshape(0, CONTEXT_LENGTH), [])

        # Get tokens for each function
        tokens = self.tokenize_functions(functions)
        function_list = list(functions.keys())

        return tokens, function_list


def flatten(lst):
    return [item for sublist in lst for item in sublist]


def collate_fn(data):
    examples = [item[0] for item in data]
    funclist = flatten([item[1] for item in data])
    return torch.cat(examples), funclist


# Small Test
# dataset = UnittestDataset("assets/filelist.json")
# dataloader = DataLoader(dataset, collate_fn=collate_fn, batch_size=2)

# for idx, batch in enumerate(dataloader, 0):
#     data, funcs = batch
#     print(data.shape)
#     print(len(funcs))
#     if idx == 0:
#         sys.exit(0)
