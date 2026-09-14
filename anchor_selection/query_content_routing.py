import torch
import json
import argparse

from pathlib import Path
from tqdm import tqdm

import torch.nn.functional as F

from transformers import DebertaV2Tokenizer, DebertaV2Model

batch_size = 1

def split_batch(data, size):
    return [
        data[i : i + size]
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

            mask = inputs["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()
            hidden = hidden * mask

            emb = hidden.sum(1) / mask.sum(1)

            embeddings.append(emb.cpu())

    return torch.cat(embeddings, dim=0)


def select_content_cluster(query_emb, centers):
    query_emb = F.normalize(query_emb.unsqueeze(0), dim=1)
    centers = F.normalize(centers, dim=-1)

    similarity = torch.matmul(query_emb, centers.T).squeeze(0)

    cid = torch.argmax(similarity).item()
    score = similarity[cid].item()

    # 保存每个 cluster 的 score，保留三位小数
    all_scores = {
        str(i): round(similarity[i].item(), 3)
        for i in range(similarity.size(0))
    }

    return cid, score, all_scores

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test_path",
        default="../data/tweet_paraphrase/test_100/user_test_100.json",
        type=str
    )

    parser.add_argument(
        "--user_group_path",
        default="./tweet_paraphrase/user_group_routing.json"
    )

    parser.add_argument(
        "--content_root",
        default="./tweet_paraphrase/group_content"
    )

    parser.add_argument(
        "--output_path",
        default="./tweet_paraphrase/query_content_scores.json"
    )


    args=parser.parse_args()

    with open(args.test_path, "r", encoding="utf-8") as f:
        users = json.load(f)

    with open(args.user_group_path, "r", encoding="utf-8") as f:
        user_group = json.load(f)

    tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-large")
    model = DebertaV2Model.from_pretrained("microsoft/deberta-v3-large").cuda()

    model.eval()

    results = {}

    for user in tqdm(users):
        user_id = str(user["user_id"])

        if user_id not in user_group:
            print(f"missing user: {user_id}")
            continue

        group_id = user_group[user_id]["group_id"]
        center_path = Path(args.content_root) / f"group_{group_id}" / "content_centers.pt"

        if not center_path.exists():
            print(f"missing center: {center_path}")
            continue

        content_centers = torch.load(center_path)

        queries = user["query"]
        query_texts = [q["input"] for q in queries]

        query_ids = [q["id"] for q in queries]

        query_embeddings = encode_query(
            query_texts,
            tokenizer,
            model
        )

        for idx, qid in enumerate(query_ids):
            cid, score, all_scores = select_content_cluster(query_embeddings[idx], content_centers)
            results[str(qid)] = {
                "user_id": int(user_id),
                "group_id": group_id,
                "content_cluster_id": cid,
                "similarity": score,
                "content_cluster_scores": all_scores
            }

    output_path = Path(args.output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"Saved: {output_path}")

if __name__=="__main__":
    main()
