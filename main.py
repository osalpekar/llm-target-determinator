import json
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.optim as optim

from transformers import BertTokenizer, BertModel, BertConfig


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
        self.tokenizer = AutoTokenizer.from_pretrained(model_type)
        self.test_index = None

        # TODO: need to figure out how to handle this. It will allow us to map
        # back from indices in the embedding matrix to testfiles
        self.index_to_testfile = {}

    def encode_batch(self, batched_input, debug=False):
        # tokenized_input = self.tokenizer(
        #     batched_input,
        #     return_tensors='pt',
        #     padding=True
        # )

        if debug:
            print(tokenized_input)
            print(self.tokenizer.decode(tokenized_input['input_ids'][0]))

        out = self.model(**tokenized_input)
        return out[1]
    
    def index(self, tokens_file):
        embedding_list = []

        with open(tokens_file) as f:
            data = json.load(f)

        for idx, filename in enumerate(data):
            embeddings = self.encode_batch(data[filename], debug=False)
            embeddings_summed = torch.sum(embeddings, dim=0).reshape(1, 768)
            embedding_list.append(embeddings_summed)
            self.index_to_testfile[idx] = filename

        self.test_index = torch.cat(embedding_list, dim=0)


class TwoTower:
    def __init__(self, all_data):
        self.indexer = Indexer()
        self.indexer.index(tokens_file)

    def predict(self, diff):
        diff_embedding = self.indexer.encode_batch(batched_input)
        
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
    self.indexer = Indexer()
    self.indexer.index(args.code_tokens)


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
