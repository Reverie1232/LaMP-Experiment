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


batch_size = 4


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
    default="../data/scholarly_title/user_anchor_candidate.json",
    type=str
)


parser.add_argument(
    "--task_name",
    default="scholarly_title",
    type=str
)


parser.add_argument(
    "--k",
    default=150,
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

all_interactions = []
for user_idx, user in enumerate(tqdm(anchor_candidate)):
    for history_idx, p in enumerate(user['profile']):
        new_p = {}
        for k,v in p.items():
            if isinstance(v,str):
                new_p[k]=get_first_k_tokens(v,368)
            else:
                new_p[k]=str(v)
        text = prompt_template[args.task_name]["retrieval_history"].format(**new_p)
        all_interactions.append({
            "user_idx": user_idx,
            "history_idx": history_idx,
            "text": text
        })

all_interaction_emb = []
for start in tqdm(range(0, len(all_interactions), batch_size)):
    batch = all_interactions[start:start+batch_size]
    texts = [x["text"] for x in batch]
    with torch.no_grad():
        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to("cuda")
        outputs = model(**inputs)
        hidden_states = outputs.last_hidden_state
        attention_mask = (inputs["attention_mask"].unsqueeze(-1).expand(hidden_states.size()).float())
        hidden_states = hidden_states * attention_mask
        summed = hidden_states.sum(dim=1)
        count = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
        mean_pool = summed / count
        all_interaction_emb.append(mean_pool.cpu())
        
all_interaction_emb = torch.cat(all_interaction_emb, dim=0)
print(f"All interaction embedding shape: {all_interaction_emb.shape}")

save_path = Path(f'./{args.task_name}/all_interaction')
save_path.mkdir(parents=True, exist_ok=True)
torch.save(all_interaction_emb, save_path / 'all_interaction_emb.pt')

emb = all_interaction_emb.numpy()
print("Start clustering...")
kmeans = KMeans(n_clusters=args.k, random_state=0, max_iter=3000, n_init="auto")
labels = kmeans.fit_predict(emb)
cluster_centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float)

torch.save(cluster_centers, save_path / 'cluster_centers.pt')
torch.save(labels, save_path / 'labels.pt')

print(f"Cluster centers shape: {cluster_centers.shape}")
cluster_dir = save_path / 'clusters'
cluster_dir.mkdir(parents=True, exist_ok=True)
for cluster_id in range(args.k):
    interaction_indices = np.where(labels == cluster_id)[0]
    cluster_info = {
        "cluster_id": cluster_id,
        "num_interactions": len(interaction_indices),
        "interaction_indices": interaction_indices.tolist(),
        "user_indeces": list(set([all_interactions[idx]["user_idx"] for idx in interaction_indices]))
    }

    with open(cluster_dir / f'cluster_{cluster_id}.json', 'w', encoding='utf-8') as f:
        json.dump(cluster_info, f, ensure_ascii=False, indent=4)
    
    print(f"Cluster {cluster_id}: {len(interaction_indices)} interactions, {len(cluster_info['user_indeces'])} users")