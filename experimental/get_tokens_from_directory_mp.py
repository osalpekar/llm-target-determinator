
def process_file(file, repo_dir, tests_only):
    return get_tokens_from_file(file_path=file, repo_dir=repo_dir, tests_only=tests_only)

def get_tokens_from_directory_with_multiprocessing(directory: Path, repo_dir: Path = None, file_prefix="", tests_only=True, output_file: Optional[str] = None):
    directory = get_effective_directory(repo_dir, directory)

    print("Traversing files under {directory}...")
    all_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if should_process_file(file, root, file_prefix):
                file_path = os.path.join(root, file)
                all_files.append(file_path)
    print(f"Found {len(all_files)} files to parse under {directory}.")
    process_file_args = [(file, repo_dir, tests_only) for file in all_files]
    pool = multiprocessing.Pool()
    results = pool.starmap(process_file, process_file_args)

    all_tokens = defaultdict(list)
    for file_tokens in results:
        all_tokens.update(file_tokens)
    if output_file:
        # dump as json
        write_token_dict_as_json(all_tokens, output_file)
    return to_json(all_tokens)

