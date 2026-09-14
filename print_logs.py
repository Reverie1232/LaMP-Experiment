from huggingface_hub import list_repo_files

files = list_repo_files(
    repo_id="Zhaoxuan/P2P_data",
    repo_type="dataset"
)

for f in files:
    print(f)