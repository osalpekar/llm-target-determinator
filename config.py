from dataclasses import dataclass

@dataclass
class TDArgs:
    model: str = "codellama"
    # model: str = "bert-base-uncased"
    model_ckpt_dir: str = "~/codellama/CodeLlama-7b-Python/"
    tokenizer_path: str = "~/codellama/CodeLlama-7b-Python/tokenizer.model"
    filelist: str = "assets/filelist.json"
    max_context_len: int = 128
    max_batch_size: int = 600
