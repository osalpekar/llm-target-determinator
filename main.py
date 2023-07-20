import json
from datetime import datetime
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from transformers import BertTokenizer, BertModel, BertConfig, AutoModelForCausalLM, AutoTokenizer
from typing import Any, Dict
from pathlib import Path

import pr_tokenization
from get_tokens_from_directory import get_tokens_from_file

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
            )#.to("cuda:0")
        self.tokenizer = pr_tokenization.PTTokenizer()
        self.test_index = None

        self.index_to_testfile_func = {}

    def get_embeddings(self, tokenized_input, debug=False):
        if debug:
            print(tokenized_input)
            print(self.tokenizer.decode(tokenized_input))

        tensor = torch.tensor(tokenized_input, device="cpu")

        embedding = self.model(
            tensor,
            output_hidden_states=True
        )

        del tensor # Free CUDA memory
        return embedding.hidden_states[-1]

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
            print(f"Iter: {idx}")
            embeddings = self.get_embeddings(test_file_to_content_mapping[filename_func], debug=False)
            dims = [i for i in range(len(embeddings.shape)-1)]
            reshape_size = embeddings.shape[-1]
            embeddings_summed = torch.sum(embeddings, dim=dims).reshape(1, reshape_size) # Why is summing necessary?
            embedding_list.append(embeddings_summed)
            self.index_to_testfile_func[idx] = filename_func

        self.test_index = torch.cat(embedding_list, dim=0)

        curr_time = datetime.now()
        torch.save(self.test_index, f"test_embeddings_{curr_time}.pt")

        with open(f"func_index_mapping_{curr_time}.json", "w") as f:
            json.dump(self.index_to_testfile_func, f)


class TwoTower:
    def __init__(self, tokens_file):
        self.indexer = Indexer()
        # self.indexer.index(tokens_file)
        self.test_embedding = torch.load("test_embeddings_autograd.pt")
        print(self.test_embedding.shape)
        with open("func_index_mapping_autograd.json") as f:
            self.mapping = json.load(f)

    def predict(self, file_changed):
        pr_tokens = get_tokens_from_file(file_changed, "/home/osalpekar/pytorch")
        key = list(pr_tokens.keys())[-2]
        print(f"Using function as query: {key}")
        pr_tokens = pr_tokens[key][0]
        print(pr_tokens)
        diff_embeddings = self.indexer.get_embeddings(pr_tokens)
        dims = [i for i in range(len(diff_embeddings.shape)-1)]
        reshape_size = diff_embeddings.shape[-1]
        embeddings_summed = torch.sum(diff_embeddings, dim=dims).reshape(1, reshape_size) # (1 x emmbed_dim)

        # Each row in the similarity_matrix corresponds to the per-test scores
        # for a given diff.

        similarity_matrix = F.cosine_similarity(embeddings_summed, self.test_embedding)
        # similarity_matrix = torch.matmul(embeddings_summed, self.test_embedding.T)
        print(similarity_matrix)
        sorted_indices = torch.argsort(similarity_matrix, descending=False)
        print(sorted_indices)
        for ind in sorted_indices:
            print(self.mapping[str(ind.item())])

        return similarity_matrix


def parse_args() -> Any:
    parser = ArgumentParser("Model Trainer")
    parser.add_argument("--code-tokens", type=str, default="", help="JSON file w PyTorch code")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    nir_model = TwoTower(args.code_tokens)
    # nir_model.predict("/home/osalpekar/pytorch/torch/distributed/fsdp/api.py")
    nir_model.predict("/home/osalpekar/pytorch/torch/autograd/gradcheck.py")


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
