import json
from datetime import datetime
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from transformers import BertTokenizer, BertModel, BertConfig, AutoModelForCausalLM, AutoTokenizer
from typing import Any, Dict, List
from pathlib import Path
from collections import defaultdict

import pr_tokenization
import get_tokens_from_directory as gt
import cache_data

MODEL_TYPE = "bigcode/starcoderplus"

class Indexer:
    def __init__(self, model_type=MODEL_TYPE):
        self.model_type = model_type
        self.tokenizer = pr_tokenization.PTTokenizer()
        self.test_index = None
        self._model = None

        self.index_to_testfile_func = {}

    @property
    def model(self):
        # Load the model lazily, so that bugs earlier in the code aren't blocked on model loading
        if self._model is None:
            print("Model isn't loaded yet. Loading now...")
            self._model = AutoModelForCausalLM.from_pretrained(self.model_type, trust_remote_code=True)

        return self._model

    @model.setter
    def model(self, model):
        # If we need to replace one index with another (usually via python console) save time on model loading
        # by reusing the one already loaded
        self._model = model

    def get_embeddings(self, tokenized_input, debug=False, cache_key=None):
        if debug:
            print(tokenized_input)
            print(self.tokenizer.decode(tokenized_input))

        tensor = torch.tensor(tokenized_input, device="cpu")

        print(f"embedding cache key is {cache_key}")
        cache = cache_data.TensorCache(cache_dir="cache", namespace="test_embeddings")
        if cache_key:
            embedding = cache.get_cache_data(cache_key)

        if embedding is None:

            full_model_states = self.model(
                tensor,
                output_hidden_states=True
            )

            embedding = full_model_states.hidden_states[-1]
            cache.save_cache_data(cache_key, embedding)

        del tensor # Free CUDA memory
        return embedding

    def index(self, tokens_file):
        """
        Compute embeddings for tokens in the index file and cache results
        """
        embedding_list = []

        # Contains a dictionary mapping test file names to their contents
        with open(tokens_file) as f:
            test_file_to_content_mapping = json.load(f)

        self.index_tokens(test_file_to_content_mapping)

    def index_paths(self, repo_dir: Path, test_paths: List[Path]):
        """
        Compute embeddings for files in the given paths and cache results

        repo_dir : the root directory of the repo you want to index
        test_paths : paths you want to index, relative to the repo root
        """
        tokens_to_index = {}
        for path in test_paths:
            full_path = repo_dir / path
            assert full_path.exists(), f"{full_path} does not exist"

            # check if path is a file
            if full_path.is_file():
                tokenizeds = gt.get_tokens_from_file(full_path, repo_dir=repo_dir, tests_only=True)
            else:
                tokenizeds = gt.get_tokens_from_directory_with_multiprocessing(full_path, repo_dir=repo_dir, file_prefix="test_", tests_only=True)
            tokens_to_index.update(tokenizeds)

        self.index_tokens(tokens_to_index)

    def index_tokens(self, test_file_to_content_mapping):
        """
        Get the embeddings for all test files and save them.  These will be compared against the PR diff embeddings later.

        This function is expected to only be called once. Repeated invocations will replace any earlier indexed data
        """
        embedding_list = []

        for idx, filename_func in enumerate(test_file_to_content_mapping):
            print(f"Iter: {idx}")
            embeddings = self.get_embeddings(test_file_to_content_mapping[filename_func], debug=False, cache_key=filename_func)
            dims = [i for i in range(len(embeddings.shape)-1)]
            reshape_size = embeddings.shape[-1]
            embeddings_summed = torch.sum(embeddings, dim=dims).reshape(1, reshape_size) # Why is summing necessary?
            embedding_list.append(embeddings_summed)
            self.index_to_testfile_func[idx] = filename_func

        self.test_index = torch.cat(embedding_list, dim=0)

        # Save indexed data. We don't seem to actually use this though
        curr_time = datetime.now()
        torch.save(self.test_index, f"test_embeddings_{curr_time}.pt")

        with open(f"func_index_mapping_{curr_time}.json", "w") as f:
            json.dump(self.index_to_testfile_func, f)


class TwoTower:
    def __init__(self, indexer):
        self.indexer = indexer

    def load_test_embeddings(self, test_embeddings_file):
        self.test_embedding = torch.load("test_embeddings_autograd.pt")
        print(self.test_embedding.shape)
        with open("func_index_mapping_autograd.json") as f:
            self.test_func_index_to_func_name_mapping = json.load(f)

    def load_test_embeddings_from_indexer(self, indexer : Indexer = None):
        if indexer:
            self.indexer = indexer

        self.test_embedding = self.indexer.test_index
        print(f"Test embedding shape: {self.test_embedding.shape}")
        self.test_func_index_to_func_name_mapping = self.indexer.index_to_testfile_func

    def predict(self, file_changed, repo_root: Path):
        pr_tokens = gt.get_tokens_from_file(file_changed, repo_root)
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
            print(self.test_func_index_to_func_name_mapping[str(ind.item())])

        return similarity_matrix


def parse_args() -> Any:
    parser = ArgumentParser("Model Trainer")
    parser.add_argument("--test-embeddings", type=str, default="test_embeddings_autograd.pt", help="Embeddings for test files cached by indexer. Either set this or test-paths")
    parser.add_argument("--repo-root", type=str, default="", help="Root of pytorch repo")
    parser.add_argument("--test-paths", type=str, default="", help="List of directories or paths to compare test file against (comma separated, path relative to repo root)")
    parser.add_argument("--file-changed", type=str, default="torch/autograd/gradcheck.py", help="File to compare against")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.repo_root:
        args.repo_root = str(Path.home() / "pytorch")

    indexer = Indexer()
    nir_model = TwoTower(indexer)

    if args.test_paths:
        test_paths = list(map(lambda p: Path(p), args.test_paths.split(",")))
        indexer.index_paths(repo_dir=Path(args.repo_root), test_paths=test_paths)

        nir_model.load_test_embeddings_from_indexer(indexer)
    else:
        # Use whatever index is hard coded into the repo already.
        nir_model.load_test_embeddings("args.test_embeddings") # Method doesn't actualy respect this param
    # nir_model.predict("/home/osalpekar/pytorch/torch/distributed/fsdp/api.py")

    nir_model.predict(str(args.repo_root) + "/" + args.file_changed, args.repo_root)


if __name__ == "__main__":
    main()


class GenericObj():
    pass

def gen_args(repo_root="", test_paths="test/autograd,test/onnx/dynamo", file_changed="torch/autograd/gradcheck.py"):
    """
    For having an args obj in the python repl
    """
    args = GenericObj()
    args.repo_root = str(Path.home() / "pytorch") if not repo_root else repo_root
    args.test_paths = test_paths
    args.file_changed = file_changed
    return args

pass

### These commands are for easy execution from the python console
### My flow is: Open a new python console, then copy/paste the commands below to setup the environment
"""
import main as m
from pathlib import Path
args = m.gen_args()
indexer = m.Indexer()
nir_model = m.TwoTower(indexer)
test_paths = list(map(lambda p: Path(p), args.test_paths.split(",")))
indexer.index_paths(repo_dir=Path(args.repo_root), test_paths=test_paths)

"""



## Tests Code
# indexer = Indexer()
# indexer.index(args.code_tokens)

# print(output.shape) # (batch_size x 768)
# Below line will not work anymore since encode_batch now just returns
# last_layer_states
# print(output[2][-1].shape) # (batch_size x seq_len x 768)

# print(indexer.test_index.shape)
