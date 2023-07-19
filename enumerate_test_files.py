from pathlib import Path
import os

def enumerate_test_files(repo_root_dir: Path):
    """
    Returns all test files in the given repo
    """
    for root, dirs, files in os.walk(repo_root_dir):
        for file in files:
            if file.endswith('.py') and file.startswith('test_'):
                file_path = os.path.join(root, file)
                yield file_path