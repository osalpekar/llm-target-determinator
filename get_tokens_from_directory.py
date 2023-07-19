import os
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

def get_tokens_from_directory(directory):
    all_tokens = defaultdict(list)
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                functions = get_function_text_from_file(file_path)
                for function_name, text in functions.items():
                    tokens = extract_tokens_from_text(text)
                    if tokens.shape[1] >= MAX_TOKENS:
                        # split tokens into chunks of MAX_TOKENS
                        tokens = torch.split(tokens, MAX_TOKENS, dim=1)
                        # for i in range(0, tokens["input_ids"].shape[1], MAX_TOKENS):
                        #     token_chunk = {
                        #         "input_ids": tokens["input_ids"][:, i:i+MAX_TOKENS],
                        #         "token_type_ids": tokens["token_type_ids"][:, i:i+MAX_TOKENS],
                        #         "attention_mask": tokens["attention_mask"][:, i:i+MAX_TOKENS]
                        #     }
                        #     all_tokens[file_path + ":" + function_name].append(token_chunk)
                    else:
                        tokens = [tokens]
                    all_tokens[file_path + ":" + function_name] = tokens
                    print(all_tokens)
    return all_tokens

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=str, default='pytorch')
    args = parser.parse_args()
    pprint.pprint(get_tokens_from_directory(args.directory))
