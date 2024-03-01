import json
import os
import sys
import time
from argparse import ArgumentParser
from datetime import datetime

import torch
from config import TDArgs
from llama import Llama

from test_dataset import (
    collate_fn,
    FileGranularityDataset,
    FunctionGranularityDataset,
)
from torch.utils.data import DataLoader, DistributedSampler

from transformers import AutoModelForCausalLM

# Maps embedding granularity options to the correct dataset classes
GRANULARITIES = {
    "FILE": FileGranularityDataset,
    "FUNCTION": FunctionGranularityDataset,
}


class Indexer:
    def __init__(self, experiment_name, embedding_granularity: str):
        assert embedding_granularity in GRANULARITIES

        self.experiment_name = experiment_name
        self.config = TDArgs()

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
        x = torch.rand(1)
        print(x.device)

        # Create DataLoader
        dataset = GRANULARITIES[embedding_granularity](self.config)
        sampler = DistributedSampler(
            dataset,
            num_replicas=self.world_size,
            rank=self.local_rank,
        )
        self.dataloader = DataLoader(
            dataset,
            collate_fn=collate_fn,
            batch_size=1,
            sampler=sampler,
        )
        print("init dataloader done")

        # Load Model
        # self.model = AutoModelForCausalLM.from_pretrained(
        #     "codellama/CodeLlama-7b-Python-hf"
        # ).to(self.device)
        generator = Llama.build(
            ckpt_dir=os.path.expanduser(self.config.model_ckpt_dir),
            tokenizer_path=os.path.expanduser(self.config.tokenizer_path),
            max_seq_len=self.config.max_context_len,
            max_batch_size=self.config.max_batch_size,
            use_kv_cache=False,
            model_parallel_size=1,
        )
        generator.model = generator.model.to(self.device)
        self.model = generator.model

    def index(self):
        embeddings = []
        function_list = []

        # self.model.eval()

        with torch.no_grad():
            for idx, batch in enumerate(self.dataloader, 0):
                print(idx)
                inputs, functions = batch

                attn_mask = inputs["attn_mask"].to(self.device)
                tokens = inputs["tokens"]

                if tokens.shape[0] == 0:
                    continue
                tokens = tokens.to(self.device)
                print(tokens.shape)
                print(attn_mask.shape)

                # TODO: Setting to None, should be reshaped before passing in.

                # TODO: make tokenizer handle pad_id
                # full_model_states = self.model(
                #     inputs, output_hidden_states=True
                # )
                _, embedding = self.model.forward(
                    tokens,
                    0,
                    output_last_hidden_state=True,
                    attn_mask=attn_mask,
                )
                del attn_mask
                del tokens
                del inputs

                # Embedding is (num_functions x context_length x 4096)
                # embedding = full_model_states.hidden_states[-1].detach()

                # Pooled Embedding is (num_functions x 4096)
                pooled_embedding = torch.sum(embedding, dim=1)
                del embedding

                embedding_cpu = pooled_embedding.to("cpu")
                del pooled_embedding
                embeddings.append(embedding_cpu)
                function_list.extend(functions)

                # if idx == 1:
                #     break

        embeddings = torch.cat(embeddings)
        print(
            f"Rank {self.local_rank} generated embeddings of size {embeddings.shape}"
        )
        self.save_index(embeddings, function_list)

    def save_index(self, embeddings, function_list):
        assets_path = os.path.join("assets", self.experiment_name)

        try:
            os.mkdir(assets_path)
        except FileExistsError as e:
            pass

        torch.save(
            embeddings, f"{assets_path}/unittest_index_{self.local_rank}.pt"
        )

        with open(
            f"{assets_path}/unittest_index_mapping_{self.local_rank}.json", "w+"
        ) as f:
            json.dump({"mapping": function_list}, f)

        if self.local_rank == 0:
            print(f"Wrote Checkpoints and Test Mapping to {assets_path}")


def main():
    parser = ArgumentParser("Indexer")
    parser.add_argument(
        "--experiment-name",
        type=str,
        required=True,
        help="Name this experiment. We will store the artifacts under assets/<experiment-name>",
    )
    parser.add_argument(
        "--granularity",
        type=str,
        required=True,
        choices=GRANULARITIES.keys(),
    )
    args = parser.parse_args()

    start = time.time()
    indexer = Indexer(
        args.experiment_name, embedding_granularity=args.granularity
    )
    indexer.index()
    end = time.time()

    print(f"Total time to generate embeddings: {end-start} seconds")


if __name__ == "__main__":
    main()
