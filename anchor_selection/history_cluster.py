import torch
import json
import argparse

from pathlib import Path
from tqdm import tqdm

import numpy as np

from sklearn.cluster import KMeans

from transformers import (
    DebertaV2Tokenizer,
    DebertaV2Model
)


batch_size = 32


with open('../prompt/prompt.json', 'r') as f:
    prompt_template = json.load(f)

def split_batch(data, size):

    return [
        data[i:i+size]
        for i in range(
            0,
            len(data),
            size
        )
    ]



def get_first_k_tokens(text,k):

    return " ".join(
        text.split()[:k]
    )



parser = argparse.ArgumentParser()


parser.add_argument(
    "--candidate_path",
    default="../data/scholarly_title/user_anchor_candidate.json"
)


parser.add_argument(
    "--task_name",
    default="scholarly_title"
)


parser.add_argument(
    "--user_cluster_dir",
    default="./scholarly_title/clusters"
)


parser.add_argument(
    "--content_k",
    default=3,
    type=int
)


args = parser.parse_args()



# =========================
# Load data
# =========================


with open(
    args.candidate_path,
    "r",
    encoding="utf-8"
) as f:

    users = json.load(f)


# =========================
# Load encoder
# =========================
tokenizer = DebertaV2Tokenizer.from_pretrained(
    "microsoft/deberta-v3-large"
)

model = DebertaV2Model.from_pretrained(
    "microsoft/deberta-v3-large"
).cuda()

model.eval()

# =========================
# Encode one history
# =========================
def encode_history(history_list):
    embeddings = []
    processed_history = []

    for h in history_list:
        new_h = {}

        for k,v in h.items():
            if isinstance(v,str):
                new_h[k] = get_first_k_tokens(v, 368)
            else:
                new_h[k] = str(v)

        processed_history.append(new_h)
    
    history_texts = [
        prompt_template[args.task_name]["retrieval_history"].format(**h)
        for h in processed_history
    ]
    # print(history_texts[:5])

    for batch in split_batch(
        history_texts,
        batch_size
    ):
        with torch.no_grad():
            inputs=tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(model.device)
            outputs=model(**inputs)
            hidden=outputs.last_hidden_state
            mask=inputs["attention_mask"]
            mask=(
                mask
                .unsqueeze(-1)
                .expand(hidden.size())
                .float()
            )
            hidden=hidden*mask
            emb=(
                hidden.sum(1)
                /
                mask.sum(1)
            )
            embeddings.append(
                emb.cpu()
            )

    return torch.cat(
        embeddings,
        dim=0
    )


# =========================
# Output dir
# =========================
output_root=Path(
    f"./{args.task_name}/group_content"
)

output_root.mkdir(
    parents=True,
    exist_ok=True
)


# =========================
# Process every user cluster
# =========================
cluster_files=list(
    Path(args.user_cluster_dir)
    .glob("cluster_*.json")
)

for cluster_file in tqdm(
    cluster_files,
    desc="User clusters"
):
    with open(
        cluster_file,
        "r"
    ) as f:

        cluster_info=json.load(f)

    group_id=cluster_info["cluster_id"]
    user_indices=cluster_info["user_indices"]

    # 保存所有history
    histories=[]
    for uid in user_indices:
        user_history=users[uid]["profile"]
        for hid,h in enumerate(user_history):
            histories.append(
                {
                    "user_idx":uid,
                    "history_idx":hid,
                    "data":h
                }
            )

    if len(histories) < args.content_k:
        print(
            f"skip group {group_id}, "
            f"only {len(histories)} histories"
        )

        continue

    # =========================
    # history embedding
    # =========================

    raw_history=[
        x["data"]
        for x in histories
    ]

    history_emb=encode_history(
        raw_history
    )

    print(
        "history embedding:",
        history_emb.shape
    )

    # =========================
    # Content clustering
    # =========================
    kmeans=KMeans(
        n_clusters=args.content_k,
        random_state=0,
        n_init="auto"
    )

    labels=kmeans.fit_predict(
        history_emb.numpy()
    )

    centers=torch.tensor(
        kmeans.cluster_centers_,
        dtype=torch.float
    )

    group_dir=output_root / f"group_{group_id}"
    group_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # 保存center
    torch.save(
        centers,
        group_dir/"content_centers.pt"
    )

    # =========================
    # Save content clusters
    # =========================
    for cid in range(args.content_k):
        idxs=np.where(
            labels==cid
        )[0]

        selected_history=[
            histories[i]
            for i in idxs
        ]

        content_data={
            "group_id":group_id,
            "content_cluster_id":cid,
            "num_history":len(selected_history),
            "history":selected_history
        }

        with open(
            group_dir /
            f"content_cluster_{cid}.json",
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(
                content_data,
                f,
                indent=4,
                ensure_ascii=False
            )

    print(
        f"group {group_id}: " 
        f"{len(histories)} histories -> "
        f"{args.content_k} clusters"
    )
    
print("Finished!")