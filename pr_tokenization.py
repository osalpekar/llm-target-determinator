from typing import Any

from transformers import AutoTokenizer
from llama.tokenizer import Tokenizer

CONTEXT_LENGTH = 100


class PTTokenizer:
    def __init__(
            self,
            model_checkpoint: str = "codellama",
            tokenizer_path: str = ""
        ):
        self.model_checkpoint = model_checkpoint 
        if self.model_checkpoint != "codellama":
            self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
            # self.tokenizer.pad_token = "<fim_pad>"
        else:
            self.tokenizer = Tokenizer(tokenizer_path)

    def encode(self, data: Any) -> Any:
        if self.model_checkpoint != "codellama":
            return self.tokenizer.encode(
                data,
                return_tensors="pt",
                padding="max_length",
                max_length=CONTEXT_LENGTH,
                truncation=True,
            )
        else:
            return self.tokenizer.encode(
                data,
                bos=True,
                eos=False,
            ) 

    def decode(self, tokenized_data: Any) -> str:
        return self.tokenizer.decode(tokenized_data)
