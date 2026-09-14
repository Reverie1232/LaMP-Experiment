import json
from collections import Counter
from pathlib import Path

json_files = [
    "./data/citation/user_base_LLM.json",
    "./data/citation/user_anchor_candidate.json",
    "./data/citation/user_reserve_10_percent.json",
    "./data/citation/test_100/user_test_100.json"
]

for file in json_files:
    print("=" * 80)
    print(f"File: {file}")

    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # item数量
    if isinstance(data, list):
        print(f"Total items: {len(data)}")

        field_counter = Counter()
        type_counter = {}

        for item in data:
            if not isinstance(item, dict):
                continue

            field_counter.update(item.keys())

            for k, v in item.items():
                if k not in type_counter:
                    type_counter[k] = type(v).__name__

        print("\nFields:")
        for field, cnt in sorted(field_counter.items()):
            print(
                f"  {field:<30} "
                f"count={cnt:<8} "
                f"type={type_counter[field]}"
            )

    elif isinstance(data, dict):
        print("JSON is a dict")

        print("\nFields:")
        for k, v in data.items():
            print(
                f"  {k:<30} "
                f"type={type(v).__name__}"
            )

    else:
        print(f"Unsupported root type: {type(data)}")

    print()