import os
import re
from transformers import BertTokenizer, AutoTokenizer
import argparse
from collections import defaultdict
import ast
from typing import Dict
import pprint
import torch
import pr_tokenization

MAX_TOKENS = 8292

def get_function_text_from_file(filename: str) -> Dict[str, str]:
    with open(filename, 'r') as file:
        content = file.read()

    module = ast.parse(content)

    functions = {}

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
                    signature = node.name + '.' + sub_node.name
                    body = ast.get_source_segment(content, sub_node)
                    functions[signature] = body

    return functions

def extract_tokens_from_text(text):
    tokenizer = pr_tokenization.PTTokenizer()
    tokens = tokenizer.encode(text)
    return tokens

def extract_text_from_file(filename):
    with open(filename, 'r') as file:
        text = file.read()
    return text

def get_tokens_from_file(file_path, tests_only=False):
    print(f"Parsing {file_path}")

    all_file_tokens = defaultdict(list)

    functions = get_function_text_from_file(file_path)
    print(f"Found {len(functions.items())} functions")
    for function_name, text in functions.items():

        # Skip unless the function_name matches the regex r/.*\.test_.*/
        if tests_only and not function_name.startswith("test") and not re.match(r'.*\.test_.*', function_name):
            print(f"Skipping {function_name} since it's not a test function")
            continue

        print(f"Extracting tokens from {function_name}")
        tokens = extract_tokens_from_text(text, function_name=function_name)
        print(f"Got {tokens.shape[1]} tokens")
        if tokens.shape[1] >= MAX_TOKENS:
            # split tokens into chunks of MAX_TOKENS
            tokens = torch.split(tokens, MAX_TOKENS, dim=1)
        else:
            tokens = [tokens]
        all_file_tokens[file_path + ":" + function_name] = tokens

    return all_file_tokens

def get_tokens_from_directory(directory, file_prefix):
    all_tokens = defaultdict(list)
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py') and file.startswith(file_prefix):
                file_path = os.path.join(root, file)
                file_tokens = get_tokens_from_file(file_path, tests_only=True)
                all_tokens.update(file_tokens)
                print(f"Done parsing {file_path}")
    return all_tokens

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=str, default='pytorch')
    parser.add_argument('--file_prefix', type=str, default='test_')
    args = parser.parse_args()
    pprint.pprint(get_tokens_from_directory(args.directory, file_prefix=args.file_prefix))
