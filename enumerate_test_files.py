import os
from pathlib import Path


def enumerate_test_files(repo_root_dir: Path, prefix: str = "test_"):
    """
    Returns all test files in the given repo
    """
    for root, dirs, files in os.walk(repo_root_dir):
        for file in files:
            if file.endswith(".py") and file.startswith(prefix):
                file_path = os.path.join(root, file)
                yield file_path


def tokenize_test_files(repo_root_dir: Path):
    """
    Returns a dictionary mapping test file names to their contents
    """
    test_files = enumerate_test_files(repo_root_dir)
    test_file_to_content_mapping = {}
    for test_file in test_files:
        with open(test_file) as f:
            test_file_to_content_mapping[test_file] = f.read()
    return test_file_to_content_mapping
