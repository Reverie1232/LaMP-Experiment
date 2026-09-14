import torch
import json
import argparse
from tqdm import tqdm
from pathlib import Path
import numpy as np

from transformers import (
    DebertaV2Tokenizer,
    DebertaV2Model
)

from sklearn.cluster import KMeans


batch_size = 32


def get_first_k_tokens(text, k):
    """
    Extract first k words
    """
    tokens = text.split()
    return " ".join(tokens[:k])


def split_batch(init_list, batch_size):
    """
    split list into batches
    """
    groups = []

    for i in range(0, len(init_list), batch_size):
        groups.append(
            init_list[i:i+batch_size]
        )

    return groups



parser = argparse.ArgumentParser(
    "User clustering"
)


parser.add_argument(
    "--candidate_path",
    default="../data/product_rating/user_anchor_candidate.json",
    type=str
)


parser.add_argument(
    "--task_name",
    default="product_rating",
    type=str
)


parser.add_argument(
    "--k",
    default=50,
    type=int
)


args = parser.parse_args()



# ==============================
# Load data
# ==============================

with open(
    args.candidate_path,
    "r",
    encoding="utf-8"
) as f:

    anchor_candidate = json.load(f)



with open(
    "../prompt/prompt.json",
    "r",
    encoding="utf-8"
) as f:

    prompt_template = json.load(f)



# ==============================
# Load encoder
# ==============================


tokenizer = DebertaV2Tokenizer.from_pretrained(
    "microsoft/deberta-v3-large"
)


model = DebertaV2Model.from_pretrained(
    "microsoft/deberta-v3-large"
).cuda()


model.eval()



# ==============================
# Encode user history
# ==============================


all_user_emb = []



for user_idx, user in enumerate(
        tqdm(anchor_candidate)
):

    history_embeddings = []


    # user's history
    visible_history_list = user["profile"]


    processed_history = []


    for p in visible_history_list:

        new_p = {}

        for key,value in p.items():

            if isinstance(value,str):

                new_p[key] = get_first_k_tokens(
                    value,
                    368
                )

            else:

                new_p[key] = str(value)


        processed_history.append(new_p)



    user_nl_history_list = [

        prompt_template[args.task_name]["retrieval_history"].format(**p)

        for p in processed_history

    ]


    batches = split_batch(
        user_nl_history_list,
        batch_size
    )



    for batch in batches:


        with torch.no_grad():


            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(model.device)



            outputs = model(**inputs)


            hidden_states = outputs.last_hidden_state


            attention_mask = inputs["attention_mask"]


            attention_mask = (
                attention_mask
                .unsqueeze(-1)
                .expand(hidden_states.size())
                .float()
            )


            hidden_states = (
                hidden_states
                *
                attention_mask
            )


            summed = hidden_states.sum(
                dim=1
            )


            count = torch.clamp(
                attention_mask.sum(dim=1),
                min=1e-9
            )


            mean_pool = summed / count

        history_embeddings.append(
            mean_pool.cpu()
        )
    # user representation
    #
    # mean of all history embeddings
    user_embedding = torch.cat(
        history_embeddings,
        dim=0
    ).mean(
        dim=0,
        keepdim=True
    )

    all_user_emb.append(
        user_embedding
    )

# [num_users, hidden_dim]
all_user_emb = torch.cat(
    all_user_emb,
    dim=0
)

print(
    "User embedding:",
    all_user_emb.shape
)

# ==============================
# Save user embeddings
# ==============================
save_dir = Path(
    f"./{args.task_name}"
)

save_dir.mkdir(
    parents=True,
    exist_ok=True
)

torch.save(
    all_user_emb,
    save_dir / "user_history_emb.pt"
)

# ==============================
# KMeans clustering
# ==============================
emb = all_user_emb.numpy()

print(
    "Start KMeans..."
)

kmeans = KMeans(
    n_clusters=args.k,
    random_state=0,
    max_iter=3000,
    n_init="auto"
)

labels = kmeans.fit_predict(
    emb
)

cluster_centers = torch.tensor(
    kmeans.cluster_centers_,
    dtype=torch.float
)

# ==============================
# Save clustering information
# ==============================
torch.save(
    torch.tensor(labels),
    save_dir / "cluster_labels.pt"
)

torch.save(
    cluster_centers,
    save_dir / "cluster_centers.pt"
)


print(
    "Cluster centers:",
    cluster_centers.shape
)


# ==============================
# Save cluster user index
# ==============================
cluster_dir = save_dir / "clusters"

cluster_dir.mkdir(
    parents=True,
    exist_ok=True
)


for cluster_id in range(args.k):

    user_indices = np.where(
        labels == cluster_id
    )[0].tolist()

    cluster_info = {
        "cluster_id": cluster_id,
        "num_users": len(user_indices),
        "user_indices": user_indices
    }

    with open(cluster_dir/f"cluster_{cluster_id}.json", "w", encoding="utf-8") as f:
        json.dump(
            cluster_info,
            f,
            indent=4
        )



    print(f"Cluster {cluster_id}: "f"{len(user_indices)} users")


print("Done!")