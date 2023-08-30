import os
import time
from datetime import datetime

import torch

from test_dataset import collate_fn, UnittestDataset
from torch.utils.data import DataLoader

from transformers import AutoModelForCausalLM


class Indexer:
    def __init__(self):
        # Create DataLoader
        dataset = UnittestDataset("assets/filelist.json")
        self.dataloader = DataLoader(dataset, collate_fn=collate_fn, batch_size=2)
        print("init dataloader done")

        # Init Rank/Device
        try:
            self.local_rank = os.environ["LOCAL_RANK"]
        except KeyError:
            # LOCAL_RANK may not be set if torchrun/torchx is not being used
            self.local_rank = 0

        self.device = (
            torch.device(f"cuda:{local_rank}")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        # Load Model
        self.model = AutoModelForCausalLM.from_pretrained("bert-base-uncased").to(
            self.device
        )

    def index(self):
        embeddings = []
        function_list = []
        for idx, batch in enumerate(self.dataloader, 0):
            inputs, functions = batch
            inputs = inputs.to(self.device)

            full_model_states = self.model(inputs, output_hidden_states=True)
            embedding = full_model_states.hidden_states[-1].detach()

            embeddings.append(embedding)
            function_list.extend(functions)
            # embedding and functions have the same order

            if idx == 2:
                break

        embeddings = torch.cat(embeddings)
        print(embeddings)
        print(function_list)
        self.save_index(embeddings, function_list)

    def save_index(self, embeddings, function_list):
        rand = hash(datetime.now())
        torch.save(embeddings, f"unittest_index_{rand}.pt")

        with open(f"unittest_index_mapping_{rand}.json") as f:
            json.dump({"mapping": function_list}, f)


if __name__ == "__main__":
    start = time.time()
    indexer = Indexer()
    indexer.index()
    end = time.time()

    print(f"Total time to generate embeddings: {end-start} seconds")
