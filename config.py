from dataclasses import dataclass

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent


@dataclass
class TDArgs:
    # Model Metadata Configs
    model: str = "codellama"
    model_ckpt_dir: str = str(
        REPO_ROOT.parent / "codellama/CodeLlama-7b-Python/"
    )
    tokenizer_path: str = str(
        REPO_ROOT.parent / "codellama/CodeLlama-7b-Python/tokenizer.model"
    )

    # Model Runtime Configs
    max_context_len: int = 128
    max_batch_size: int = 600

    # Test Parsing Configs
    project_dir: str = str(REPO_ROOT.parent / "pytorch")
    file_prefix: str = "test_"
