import json
from pathlib import Path
from collections import defaultdict

# ==============================
# Config
# ==============================

cluster_dir = Path("./citation/all_interaction/clusters")

# ==============================
# user -> clusters
# ==============================

user2clusters = defaultdict(set)

cluster_files = sorted(cluster_dir.glob("cluster_*.json"))

for file in cluster_files:

    with open(file, "r", encoding="utf-8") as f:
        cluster = json.load(f)

    cluster_id = cluster["cluster_id"]

    for user_id in cluster["user_indeces"]:
        user2clusters[user_id].add(cluster_id)

# ==============================
# Statistics
# ==============================

cluster_num_distribution = defaultdict(int)

multi_cluster_users = []

for user_id, clusters in user2clusters.items():

    num = len(clusters)

    cluster_num_distribution[num] += 1

    if num > 1:
        multi_cluster_users.append(
            (user_id, sorted(list(clusters)))
        )

num_users = len(user2clusters)

num_multi = len(multi_cluster_users)

print("=" * 60)
print(f"Total users: {num_users}")
print(f"Users appearing in multiple clusters: {num_multi}")
print(f"Ratio: {num_multi / num_users:.4f}")
print("=" * 60)

print("Distribution:")
for k in sorted(cluster_num_distribution):
    print(f"{k:2d} clusters : {cluster_num_distribution[k]} users")

print("=" * 60)

print("Examples (first 20):")

for user_id, clusters in multi_cluster_users[:20]:
    print(
        f"User {user_id}: {clusters}"
    )