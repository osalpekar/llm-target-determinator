import json
import os
import sys
import time
from datetime import datetime

import torch

from test_dataset import collate_fn, UnittestDataset
from torch.utils.data import DataLoader, DistributedSampler

from transformers import AutoModelForCausalLM


class Indexer:
    def __init__(self):
        # Init Rank/Device
        try:
            self.local_rank = int(os.environ["LOCAL_RANK"])
            self.world_size = int(os.environ["WORLD_SIZE"])
        except KeyError:
            # LOCAL_RANK may not be set if torchrun/torchx is not being used
            self.local_rank = 0
            self.world_size = 1

        self.device = (
            torch.device(f"cuda:{self.local_rank}")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        # Create DataLoader
        dataset = UnittestDataset("assets/filelist.json")
        sampler = DistributedSampler(
            dataset, num_replicas=self.world_size, rank=self.local_rank
        )
        self.dataloader = DataLoader(
            dataset, collate_fn=collate_fn, batch_size=2, sampler=sampler
        )
        print("init dataloader done")

        # Load Model
        self.model = AutoModelForCausalLM.from_pretrained("bert-base-uncased").to(
            self.device
        )

    def index(self):
        embeddings = []
        function_list = []

        self.model.eval()

        with torch.no_grad():
            for idx, batch in enumerate(self.dataloader, 0):
                inputs, functions = batch
                inputs = inputs.to(self.device)

                full_model_states = self.model(inputs, output_hidden_states=True)
                embedding = full_model_states.hidden_states[-1].detach()

                embedding_cpu = embedding.to("cpu")
                embeddings.append(embedding_cpu)
                function_list.extend(functions)

                del embedding
                del inputs
                del full_model_states

                # if idx == 2:
                #     break

        embeddings = torch.cat(embeddings)
        # print(embeddings)
        # print(function_list)
        self.save_index(embeddings, function_list)

    def save_index(self, embeddings, function_list):
        rand = hash(datetime.now()) & sys.maxsize
        torch.save(embeddings, f"assets/unittest_index_{rand}_{self.local_rank}.pt")

        with open(
            f"assets/unittest_index_mapping_{rand}_{self.local_rank}.json", "w+"
        ) as f:
            json.dump({"mapping": function_list}, f)


if __name__ == "__main__":
    start = time.time()
    indexer = Indexer()
    indexer.index()
    end = time.time()

    print(f"Total time to generate embeddings: {end-start} seconds")
