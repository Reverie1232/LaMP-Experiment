import os
import json

ROOT_DIR = "./scholarly_title/group_content"

count = 0

for group_id in range(50):
    group_dir = os.path.join(ROOT_DIR, f"group_{group_id}")

    if not os.path.isdir(group_dir):
        continue

    for cluster_id in range(3):
        file_path = os.path.join(
            group_dir,
            f"content_cluster_{cluster_id}.json"
        )

        if not os.path.isfile(file_path):
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        num_history = data.get("num_history")

        if num_history is not None and num_history < 10:
            count += 1
            print(f"{file_path}: num_history = {num_history}")

print(f"\n共发现 {count} 个文件的 num_history < 10")