from typing import Any
import os

import torch

from transformers import AutoTokenizer
# TODO: import this as llamaTokenizer to prevent symbol conflicts
from llama.tokenizer import Tokenizer


class PTTokenizer:
    def __init__(
            self,
            config,
        ):
        self.config = config

        if self.config.model == "codellama":
            self.tokenizer = Tokenizer(os.path.expanduser(self.config.tokenizer_path))
            self.pad_id = self.tokenizer.eos_id
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model)
            self.tokenizer.pad_token = "<fim_pad>"
            self.pad_id = self.tokenizer.pad_token

    def encode(self, data: Any) -> Any:
        if self.config.model == "codellama":
            return self.tokenizer.encode(
                data,
                bos=True,
                eos=False,
            ) 
        else:
            return self.tokenizer.encode(
                data,
                return_tensors="pt",
                padding="max_length",
                max_length=self.config.max_context_len,
                truncation=True,
            )

    def decode(self, tokenized_data: Any) -> str:
        return self.tokenizer.decode(tokenized_data)
