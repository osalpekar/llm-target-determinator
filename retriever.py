import time
import json

import torch
import torch.nn.functional as F
from glob import glob

from preproc import get_functions
from pr_tokenization import PTTokenizer
from transformers import AutoModelForCausalLM

class Retriever:
    def __init__(self):
        embeddings_files = glob("assets/unittest_index_*.pt")

        chunks = {}
        for f in embeddings_files:
            chunk_num = int(f.split(".")[0].split("_")[-1])
            split_name = f.split(".")[0].split("_")
            split_name.insert(2, "mapping")
            testlist_json = "_".join(split_name) + ".json"
            chunks[chunk_num] = [f, testlist_json] 

        print(chunks)
        self.embeddings = []
        self.unittest_names = []
        for i in range(len(embeddings_files)):
            self.embeddings.append(torch.load(chunks[i][0]))
            with open(chunks[i][1]) as f:
                test_map = json.load(f)

            self.unittest_names.extend(test_map["mapping"])

        self.embeddings = torch.cat(self.embeddings).to("cuda:0")
        print(self.embeddings.shape)
        self.tokenizer = PTTokenizer("bert-base-uncased")
        self.model = AutoModelForCausalLM.from_pretrained(
            "bert-base-uncased"
        ).to("cuda:0")

    def retrieve(self):
        # parse and tokenize input (function from a file)
        # run model forward on each chunk of the embeddings
        # cosine similarity per chunk
        # rank and print most/least match
        functions = get_functions("/home/osalpekar/pytorch/torch/distributed/fsdp/fully_sharded_data_parallel.py")
        for signature in functions:
            if "__init__" in signature:
                function_body = functions[signature]
                tokens = self.tokenizer.encode(function_body).to("cuda:0")

        self.model.eval()
        with torch.no_grad():
            full_model_states = self.model(
                    tokens, output_hidden_states=True
            )
            test_embedding = full_model_states.hidden_states[-1].detach()

            similarity_matrix = F.cosine_similarity(
                self.embeddings, test_embedding
            )
            print(similarity_matrix)
            sorted_indices = torch.argsort(similarity_matrix, descending=False)
            print(sorted_indices)
            for ind in sorted_indices:
                print(self.unittest_names[int(ind.item())])

if __name__ == "__main__":
    start = time.time()
    retriever = Retriever()
    retriever.retrieve()
    end = time.time()

    print(f"Total time to retreieve: {end-start} seconds")
