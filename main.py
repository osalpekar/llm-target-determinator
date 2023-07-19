import json
from datetime import datetime
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.optim as optim

from transformers import BertTokenizer, BertModel, BertConfig, AutoModelForCausalLM, AutoTokenizer
from typing import Any, Dict
from pathlib import Path

import pr_tokenization

text1 = "Replace me by any text you'd like."
text2 = "Hello me by any text you'd like."
text3 = "any text you'd like."
text4 = "Hello me you'd like."
all_data = [[text1, text2], [text3, text4]]


class Indexer:
    def __init__(self):
        model_type = "bigcode/starcoderplus"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_type,
            trust_remote_code=True
        )
        self.tokenizer = pr_tokenization.PTTokenizer()
        self.test_index = None

        self.index_to_testfile_func = {}

    def get_embeddings(self, dataset, debug=False):
        tokenized_input = self.tokenizer.encode(dataset)

        if debug:
            print(tokenized_input)
            print(self.tokenizer.decode(tokenized_input))

        embedding = self.model(tokenized_input)
        return embedding

    def index(self, tokens_file):
        """
        Get the embeddings for all test files and save them.  These will be compared against the PR diff embeddings later.

        This function is expected to only be called once.
        """
        embedding_list = []

        # Contains a dictionary mapping test file names to their contents
        with open(tokens_file) as f:
            test_file_to_content_mapping = json.load(f)

        for idx, filename_func in enumerate(test_file_to_content_mapping):
            embeddings = self.get_embeddings(test_file_to_content_mapping[filename_func], debug=False)
            embeddings_summed = torch.sum(embeddings, dim=0).reshape(1, 768) # Why is summing necessary?
            embedding_list.append(embeddings_summed)
            self.index_to_testfile_func[idx] = filename_func

        self.test_index = torch.cat(embedding_list, dim=0)

        curr_time = datetime.now()
        torch.save(self.test_index, f"test_embeddings_{curr_time}.pt")

        with open(f"func_index_mapping_{curr_time}.json", "w") as f:
            json.dump(self.index_to_testfile_func, f)


class TwoTower:
    def __init__(self, all_data):
        self.indexer = Indexer()
        self.indexer.index(tokens_file)

    def predict(self, diff):
        diff_embedding = self.indexer.get_embeddings(batched_input)

        # Each row in the similarity_matrix corresponds to the per-test scores
        # for a given diff.
        similarity_matrix = torch.matmul(diff_embedding, self.indexer.test_index.T)

        return similarity_matrix


def parse_args() -> Any:
    parser = ArgumentParser("Model Trainer")
    parser.add_argument("--code-tokens", type=str, help="JSON file w PyTorch code")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    indexer = Indexer()
    indexer.index(args.code_tokens)


if __name__ == "__main__":
    main()

## Tests Code
# indexer = Indexer()
# output = indexer.encode_batch([text1, text2])

# print(output.shape) # (batch_size x 768)
# Below line will not work anymore since encode_batch now just returns
# last_layer_states
# print(output[2][-1].shape) # (batch_size x seq_len x 768)

# indexer.index(all_data)
# print(indexer.test_index.shape)
