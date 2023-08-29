import time

from transformers import AutoModelForCausalLM
from torch.utils.data import DataLoader
from test_dataset import TestDataset, collate_fn

class Indexer:
    def __init__(self):
        # Create DataLoader
        dataset = TestDataset("assets/filelist.json")
        self.dataloader = DataLoader(dataset, collate_fn=collate_fn, batch_size=2)
        print("init dataloader done")

        # Load Model
        self.model = AutoModelForCausalLM.from_pretrained(
            'bert-base-uncased'
        )

    def index(self):
        for idx, batch in enumerate(self.dataloader, 0):
            full_model_states = self.model(batch, output_hidden_states=True)
            embedding = full_model_states.hidden_states[-1].detach()


if __name__ == "__main__":
    start = time.time()
    indexer = Indexer()
    indexer.index()
    end = time.time()

    print(f"Total time to generate embeddings: {end-start} seconds")
