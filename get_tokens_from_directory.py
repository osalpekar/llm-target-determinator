import os
from transformers import BertTokenizer
import argparse
import defaultdict

def extract_tokens_from_text(text):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokens = tokenizer.tokenize(text)
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
                text = extract_text_from_file(file_path)
                tokens = extract_tokens_from_text(text)
                if len(tokens) >= 8192:
                    # split tokens into chunks of 8192
                    tokens = [tokens[i:i+8192] for i in range(0, len(tokens), 8192)]
                else:
                    tokens = [tokens]
                all_tokens[file_path] = tokens
                break
    return all_tokens

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=str, default='pytorch')
    args = parser.parse_args()
    print(get_tokens_from_directory(args.directory))
