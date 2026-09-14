import torch
import json
import argparse
from tqdm import tqdm
from pathlib import Path

import torch.nn.functional as F

from transformers import DebertaV2Tokenizer, DebertaV2Model

batch_size = 32

def get_first_k_tokens(text, k):
    tokens = text.split()
    return " ".join(tokens[:k])

def split_batch(init_list, batch_size):
    return [
        init_list[i : i + batch_size]
        for i in range(0, len(init_list), batch_size)
    ]

def encode_user_history(history_list, tokenizer, model, prompt_template, task_name):
    """
    Encode user's history
    """
    processed_history = []

    for p in history_list:
        new_p = {}
        for key, value in p.items():
            if isinstance(value, str):
                new_p[key] = get_first_k_tokens(value, 368)
            else:
                new_p[key] = str(value)

        processed_history.append(new_p)

    history_texts = [
        prompt_template[task_name]["retrieval_history"].format(**p)
        for p in processed_history
    ]

    history_embeddings = []

    for batch in split_batch(history_texts, batch_size):
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
            attention_mask = inputs["attention_mask"].unsqueeze(-1).expand(hidden_states.size()).float()
            hidden_states = hidden_states * attention_mask
            summed = hidden_states.sum(dim=1)
            count = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
            
            mean_pool = summed / count
            history_embeddings.append(mean_pool.cpu())

        user_embedding = torch.cat(history_embeddings, dim=0).mean(dim=0)

        return user_embedding


def select_group(user_embedding, cluster_centers):
    user_embedding = F.normalize(user_embedding.unsqueeze(0), dim=-1)
    cluster_centers = F.normalize(cluster_centers, dim=-1)

    similarity = torch.matmul(user_embedding, cluster_centers.T)

    group_id = torch.argmax(similarity, dim=1).item()

    score = similarity[0, group_id].item()

    return group_id, score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target_user_path",
        default="../data/scholarly_title/test_100/user_test_100.json",
        type=str
    )

    parser.add_argument(
        "--cluster_center_path",
        default="./scholarly_title/cluster_centers.pt",
        type=str
    )

    parser.add_argument(
        "--task_name",
        default="scholarly_title",
        type=str
    )

    parser.add_argument(
        "--output_path",
        default="./scholarly_title/user_group_routing.json",
        type=str
    )

    args = parser.parse_args()

    with open(args.target_user_path, "r", encoding="utf-8") as f:
        users = json.load(f)
        print(f"Total users: {len(users)}")

    with open("../prompt/prompt.json", "r", encoding="utf-8") as f:
        prompt_template = json.load(f)


    tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-large")
    model = DebertaV2Model.from_pretrained("microsoft/deberta-v3-large").cuda()

    model.eval()


    cluster_centers = torch.load(args.cluster_center_path)
    print(f"Cluster centers: {cluster_centers.shape}")

    results = {}

    for user in tqdm(users):
        user_id = user["user_id"]
        history = user["profile"]

        user_embedding = encode_user_history(
            history,
            tokenizer,
            model,
            prompt_template,
            args.task_name
        )

        group_id, score = select_group(
            user_embedding,
            cluster_centers
        )

        results[str(user_id)] = {
            "group_id": group_id,
            "similairy": score
        }


    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            results,
            f,
            indent=4,
            ensure_ascii=False
        )

    print(f"Saved: {output_path}")

if __name__=="__main__":
    main()
