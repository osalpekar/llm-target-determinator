import json
import os

from torch.utils.data import Dataset
from pr_tokenization import PTTokenizer

class TestDataset(Dataset):
    def __init__(self, filelist):
        with open(filelist) as f:
            filelist_json = json.load(f)

        self.filelist = filelist_json["all_files"]
        self.tokenizer = PTTokenizer()

    def __len(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        # parse ast and get functions
        # get function text
        # tokenize function
        filename = self.filelist[idx]
        with open(filename) as f:
            content = f.read()

        module = ast.parse(content)

        functions = {}

        # TODO: move this to offline
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
                        signature = nod.name + "." + sub_node.name
                        body = ast.get_source_segment(content, sub_node)
                        functions[signature] = body
        
        for signature in functions:
            function_body = functions[signature]
            tokens = self.tokenizer.encode(function_body)

        return tokens


