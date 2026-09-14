import torch
import json
import argparse

from pathlib import Path
from tqdm import tqdm

import torch.nn.functional as F

from transformers import DebertaV2Tokenizer, DebertaV2Model


batch_size = 32

def split_batch(data, size):
    return [
        data[i: i + size]
        for i in range(0, len(data), size)
    ]


def get_first_k_tokens(text, k):
    return " ".join(text.split()[:k])


def encode_query(query_list, tokenizer, model):
    processed = []

    for q in query_list:
        processed.append(get_first_k_tokens(q, 368))

    embeddings = []

    for batch in split_batch(processed, batch_size):
        with torch.no_grad():
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(model.device)

            outputs = model(**inputs)
            hidden = outputs.last_hidden_state

            # Mean Pooling
            mask = (
                inputs["attention_mask"]
                .unsqueeze(-1)
                .expand(hidden.size())
                .float()
            )

            hidden = hidden * mask

            emb = hidden.sum(1) / mask.sum(1)

            embeddings.append(emb.cpu())

    return torch.cat(embeddings, dim=0)


def select_preference_cluster(query_emb, centers):
    """
    query_emb:
        [hidden_dim]

    centers:
        [num_clusters, hidden_dim]
        num_clusters = 150

    return:
        cluster_id
        similarity
        all_scores
    """

    # [1, hidden_dim]
    query_emb = F.normalize(
        query_emb.unsqueeze(0),
        dim=1
    )

    # [150, hidden_dim]
    centers = F.normalize(
        centers,
        dim=-1
    )

    # [1, 150] -> [150]
    similarity = torch.matmul(
        query_emb,
        centers.T
    ).squeeze(0)

    # Top-1 cluster
    cid = torch.argmax(similarity).item()

    score = similarity[cid].item()

    # 保存全部 150 个 cluster 的 similarity
    all_scores = {
        str(i): round(similarity[i].item(), 3)
        for i in range(similarity.size(0))
    }

    return cid, score, all_scores


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--test_path",
        default="../data/scholarly_title/test_100/user_test_100.json",
        type=str
    )

    # 现在只有一个全局 centers.pt
    parser.add_argument(
        "--center_path",
        default="./scholarly_title/all_interaction/cluster_centers.pt",
        type=str
    )

    parser.add_argument(
        "--output_path",
        default="./scholarly_title/all_interaction/query_preference_scores.json",
        type=str
    )

    args = parser.parse_args()


    # ============================================================
    # 1. Load test users
    # ============================================================

    with open(args.test_path, "r", encoding="utf-8") as f:
        users = json.load(f)


    # ============================================================
    # 2. Load global preference centers
    # ============================================================

    center_path = Path(args.center_path)

    if not center_path.exists():
        raise FileNotFoundError(
            f"Missing center file: {center_path}"
        )

    centers = torch.load(
        center_path,
        map_location="cpu"
    )

    # 防止保存的是 dict
    if isinstance(centers, dict):
        if "centers" in centers:
            centers = centers["centers"]
        else:
            raise ValueError(
                "centers.pt is a dict, but key 'centers' was not found."
            )

    centers = centers.float()

    print("Loaded centers:", centers.shape)

    assert centers.dim() == 2, \
        f"Expected centers [num_clusters, hidden_dim], got {centers.shape}"

    assert centers.size(0) == 150, \
        f"Expected 150 clusters, got {centers.size(0)}"


    # ============================================================
    # 3. Load DeBERTa
    # ============================================================

    tokenizer = DebertaV2Tokenizer.from_pretrained(
        "microsoft/deberta-v3-large"
    )

    model = DebertaV2Model.from_pretrained(
        "microsoft/deberta-v3-large"
    ).cuda()

    model.eval()


    # ============================================================
    # 4. Routing
    # ============================================================

    results = {}

    for user in tqdm(users):

        user_id = str(user["user_id"])

        queries = user["query"]

        query_texts = [
            q["input"]
            for q in queries
        ]

        query_ids = [
            q["id"]
            for q in queries
        ]


        # --------------------------------------------------------
        # Encode all queries of this user
        # --------------------------------------------------------

        query_embeddings = encode_query(
            query_texts,
            tokenizer,
            model
        )


        # --------------------------------------------------------
        # Route every query to 150 preference clusters
        # --------------------------------------------------------

        for idx, qid in enumerate(query_ids):

            cid, score, all_scores = select_preference_cluster(
                query_embeddings[idx],
                centers
            )

            results[str(qid)] = {
                "user_id": int(user_id),

                # 全局 preference cluster
                "preference_cluster_id": cid,

                # top-1 similarity
                "similarity": score,

                # query -> 150 preference clusters
                "preference_cluster_scores": all_scores
            }


    # ============================================================
    # 5. Save
    # ============================================================

    output_path = Path(args.output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            indent=4,
            ensure_ascii=False
        )

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()