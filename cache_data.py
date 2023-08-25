from pathlib import Path

import torch

class TensorCache:
    def __init__(self, cache_dir: Path, namespace: str):
        if isinstance(cache_dir, str):
            cache_dir = Path(cache_dir)

        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.namespace = namespace

    def get_cache_file(self, key):
        key = str(key).replace("/","_") # don't let the key look like a path, even if it was derived from one
        return self.cache_dir / f"{self.namespace}_{key}.pt"

    def get_cache_data(self, key):
        cache_file = self.get_cache_file(key)
        if cache_file.is_file():
            return torch.load(cache_file)
        else:
            return None

    def save_cache_data(self, key, data):
        cache_file = self.get_cache_file(key)
        torch.save(data, cache_file)
