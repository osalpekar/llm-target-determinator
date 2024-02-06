import argparse
import json
import os

from pathlib import Path

import numpy as np

# This program is the only offline component in the indexing pipeline. This
# script is run once to generate the list of Python test files and write this
# to a filelist. This filelist is read during indexing. The parsing and
# tokenization of the functions happens in batches in the dataloader of the
# indexer.


def should_include_file(file: str, root: str, file_prefix: str) -> bool:
    """
    Returns whether the specified file should be included in the file list.
    Note that we are only interested in Python files with the specified prefix,
    provided those files do not lie in certain directories like third_party.
    """
    return (
        file.endswith(".py")
        and file.startswith(file_prefix)
        and "third_party" not in root
    )


def create_filelist(project_dir: str, file_prefix: str) -> dict[str, list[str]]:
    """
    Returns the list of files of interest in the specified subdirectory.
    """
    all_files = []
    project_dir = Path(project_dir).expanduser()

    for root, dirs, files in os.walk(project_dir):
        for file in files:
            if should_include_file(file, root, file_prefix):
                file_path = os.path.join(root, file)
                all_files.append(file_path)

    return {"all_files": all_files}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-dir",
        type=str,
        default="~/pytorch",
        help="Root Directory of the project for which we will generate the file list",
    )
    parser.add_argument(
        "--file-prefix",
        type=str,
        default="test_",
        help="Only keep files that begin with this prefix",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="assets/filelist.json",
        help="Write the produced file list to the specified output file",
    )
    args = parser.parse_args()

    file_list = create_filelist(args.project_dir, args.file_prefix)

    if args.output_file:
        with open(args.output_file, "w+") as f:
            json.dump(file_list, f)
