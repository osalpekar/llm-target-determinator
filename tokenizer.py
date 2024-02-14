import os
from typing import Any

import torch

from llama.tokenizer import Tokenizer as LlamaTokenizer

from transformers import AutoTokenizer


class Tokenizer:
    def __init__(
        self,
        config,
    ):
        self.config = config

        if self.config.model == "codellama":
            self.tokenizer = LlamaTokenizer(
                os.path.expanduser(self.config.tokenizer_path)
            )
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
