import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from tqdm import tqdm

from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)

from transformers import (
    DebertaV2Tokenizer,
    DebertaV2Model
)

batch_size = 32


# ===============================
# Utils
# ===============================

def split_batch(data, size):
    return [data[i:i+size] for i in range(0, len(data), size)]


def get_first_k_tokens(text, k):
    return " ".join(text.split()[:k])


# ===============================
# Args
# ===============================

parser = argparse.ArgumentParser()

parser.add_argument("--candidate_path", default="../data/product_rating/user_anchor_candidate.json")
parser.add_argument("--task_name", default="product_rating")
parser.add_argument("--user_cluster_dir", default="./product_rating/clusters")
parser.add_argument("--max_k", type=int, default=6)

args = parser.parse_args()


# ===============================
# Prompt
# ===============================

with open("../prompt/prompt.json") as f:
    prompt_template = json.load(f)

with open(args.candidate_path) as f:
    users = json.load(f)


# ===============================
# Encoder
# ===============================

tokenizer = DebertaV2Tokenizer.from_pretrained(
    "microsoft/deberta-v3-large"
)

model = DebertaV2Model.from_pretrained(
    "microsoft/deberta-v3-large"
).cuda()

model.eval()


# ===============================
# Encode
# ===============================

def encode_history(history_list):

    processed = []

    for h in history_list:

        new_h = {}

        for k,v in h.items():

            if isinstance(v,str):
                new_h[k]=get_first_k_tokens(v,368)
            else:
                new_h[k]=str(v)

        processed.append(new_h)

    texts=[
        prompt_template[args.task_name]["retrieval_history"].format(**x)
        for x in processed
    ]

    emb=[]

    for batch in split_batch(texts,batch_size):

        with torch.no_grad():

            inputs=tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(model.device)

            out=model(**inputs)

            hidden=out.last_hidden_state

            mask=inputs["attention_mask"]

            mask=mask.unsqueeze(-1).float()

            mean=(hidden*mask).sum(1)/mask.sum(1)

            emb.append(mean.cpu())

    return torch.cat(emb,0).numpy()


# ===============================
# Analysis
# ===============================

cluster_files=list(
    Path(args.user_cluster_dir).glob("cluster_*.json")
)

all_results=[]

for cluster_file in tqdm(cluster_files):

    with open(cluster_file) as f:

        group=json.load(f)

    histories=[]

    for uid in group["user_indices"]:

        histories.extend(
            users[uid]["profile"]
        )

    if len(histories)<args.max_k:
        continue

    emb=encode_history(histories)

    variance_k1=None

    for k in range(1,args.max_k+1):

        kmeans=KMeans(
            n_clusters=k,
            random_state=0,
            n_init="auto"
        )

        labels=kmeans.fit_predict(emb)

        centers=kmeans.cluster_centers_

        within_var=0

        intra=[]

        for cid in range(k):

            pts=emb[labels==cid]

            if len(pts)==0:
                continue

            diff=pts-centers[cid]

            within_var+=np.sum(diff**2)

            intra.append(
                np.mean(
                    np.linalg.norm(diff,axis=1)
                )
            )

        if variance_k1 is None:
            variance_k1=within_var

        reduction=1-within_var/variance_k1

        if k>1:

            sil=silhouette_score(
                emb,
                labels
            )

            db=davies_bouldin_score(
                emb,
                labels
            )

            ch=calinski_harabasz_score(
                emb,
                labels
            )

            center_dist=[]

            for i in range(k):
                for j in range(i+1,k):

                    center_dist.append(
                        np.linalg.norm(
                            centers[i]-centers[j]
                        )
                    )

            inter=np.mean(center_dist)

            sep=inter/np.mean(intra)

        else:

            sil=np.nan
            db=np.nan
            ch=np.nan
            sep=np.nan

        all_results.append({

            "group":group["cluster_id"],

            "k":k,

            "within_variance":within_var,

            "variance_reduction":reduction,

            "silhouette":sil,

            "davies_bouldin":db,

            "calinski_harabasz":ch,

            "separation_ratio":sep

        })


# ===============================
# Save
# ===============================

df=pd.DataFrame(all_results)

df.to_csv(
    "preference_structure.csv",
    index=False
)

summary=df.groupby("k").mean(numeric_only=True)

print(summary)

summary.to_csv(
    "preference_structure_summary.csv"
)