from dataclasses import dataclass

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent


@dataclass
class TDArgs:
    model: str = "codellama"
    # model: str = "bert-base-uncased"
    model_ckpt_dir: str = str(
        REPO_ROOT.parent / "codellama/CodeLlama-7b-Python/"
    )
    tokenizer_path: str = str(
        REPO_ROOT.parent / "codellama/CodeLlama-7b-Python/tokenizer.model"
    )
    filelist: str = "assets/filelist.json"
    max_context_len: int = 128
    max_batch_size: int = 600
