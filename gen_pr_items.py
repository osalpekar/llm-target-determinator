"""Various ways to generate strings from a PR on which to embed and compare to
test embeddings"""
import os
import subprocess
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parent


def get_merge_base() -> str:
    default_branch = f"origin/{os.environ.get('GIT_DEFAULT_BRANCH', 'main')}"
    merge_base = (
        subprocess.check_output(
            ["git", "merge-base", default_branch, "HEAD"],
            cwd=REPO_ROOT.parent / "pytorch",
        )
        .decode()
        .strip()
    )

    head = (
        subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT.parent / "pytorch"
        )
        .decode()
        .strip()
    )

    base_commit = merge_base
    if base_commit == head:
        # We are on the default branch, so check for changes since the last commit
        base_commit = "HEAD^"
    return base_commit


def query_changed_files() -> List[str]:
    base_commit = get_merge_base()

    proc = subprocess.run(
        ["git", "diff", "--name-only", base_commit, "HEAD"],
        cwd=REPO_ROOT.parent / "pytorch",
        capture_output=True,
        check=False,
    )

    if proc.returncode != 0:
        raise RuntimeError("Unable to get changed files")

    lines = proc.stdout.decode().strip().split("\n")
    lines = [line.strip() for line in lines]
    items = []
    for file in lines:
        with open(REPO_ROOT.parent / "pytorch" / file) as f:
            items.append(f.read())
    return items


def get_git_diff():
    base_commit = get_merge_base()

    proc = subprocess.run(
        ["git", "diff", base_commit, "HEAD"],
        cwd=REPO_ROOT.parent / "pytorch",
        capture_output=True,
        check=False,
    )

    if proc.returncode != 0:
        raise RuntimeError("Unable to get git diff")

    lines = proc.stdout.decode().strip()

    return [lines]


PR_ITEMS = {"GITDIFF": get_git_diff, "CHANGEDFILES": query_changed_files}
