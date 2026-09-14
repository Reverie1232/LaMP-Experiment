import json
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
import random
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from utils import (
    split_batch,
    get_first_k_tokens,
    extract_citation_title,
    extract_option,
    extract_movie,
    extract_news_cat,
    extract_news_headline,
    extract_product_review,
    extract_scholarly_title,
    extract_tweet_paraphrasing
)

parser = argparse.ArgumentParser()

parser.add_argument(
    "--model_name",
    default="mistralai/Mistral-7B-Instruct-v0.2",
)

parser.add_argument(
    "--embedding_model",
    default="BAAI/bge-small-en-v1.5",
)

parser.add_argument(
    "--task_name",
    type=str,
    default="movie_tagging",
)

parser.add_argument(
    "--batch_size",
    type=int,
    default=1,
)

parser.add_argument(
    "--persona_num",
    type=int,
    default=3,
)

parser.add_argument(
    "--max_history_tokens",
    type=int,
    default=368,
)

args = parser.parse_args()

if args.task_name == "movie_tagging":
    extract_article = extract_movie
elif args.task_name == "news_categorize":  
    extract_article = extract_news_cat
elif args.task_name == "news_headline":
    extract_article = extract_news_headline
elif args.task_name == "product_rating":
    extract_article = extract_product_review
elif args.task_name == "scholarly_title":
    extract_article = extract_scholarly_title
elif args.task_name == "tweet_paraphrase":  
    extract_article = extract_tweet_paraphrasing

with open("./prompt/prompt.json") as f:
    prompt_template = json.load(f)

with open("./prompt/prompt_profile.json") as f:
    prompt_profile_template = json.load(f)

with open(f"./data/{args.task_name}/test_100/user_test_100.json") as f:
    test_data = json.load(f)

tokenizer = AutoTokenizer.from_pretrained(
    args.model_name,
    padding_side="left",
)

tokenizer.pad_token = "[PAD]"
tokenizer.eos_token = "</s>"

llm = AutoModelForCausalLM.from_pretrained(
    args.model_name,
    device_map="auto",
    torch_dtype=torch.float16,
)

llm.config.pad_token_id = tokenizer.pad_token_id
llm.config.eos_token_id = tokenizer.eos_token_id

embedding_model = SentenceTransformer(
    args.embedding_model,
    device="cuda"
)

persona_database = {}

print("Building persona database...")

for user in tqdm(test_data):
    user_id = user["user_id"]
    history = user["profile"]
    history_texts = []
    for p in history:
        for key, value in p.items():
            if isinstance(value, str):
                p[key] = get_first_k_tokens(p[key], args.max_history_tokens)
            else:
                p[key] = str(p[key])
        history_texts.append(prompt_profile_template[args.task_name]['retrieval_history'].format(**p))

    history_embeddings = embedding_model.encode(
        history_texts,
        normalize_embeddings=True,
        batch_size=32, 
        convert_to_numpy=True,
        show_progress_bar=False
    )

    persona_num = min(args.persona_num, len(history_embeddings))
    kmeans = KMeans(n_clusters=persona_num, random_state=0, n_init=10)
    labels = kmeans.fit_predict(history_embeddings)
    centers = kmeans.cluster_centers_

    persona_prompts = []
    persona_histories = []

    for pid in range(persona_num):
        idx = np.where(labels == pid)[0]

        # 最多保留k条
        # 如果超过10条，选距离中心最近的10条
        if len(idx) > 10:
            cluster_emb = history_embeddings[idx]
            center = centers[pid]
            dist = np.linalg.norm(
                cluster_emb - center,
                axis=1
            )
            top_idx = np.argsort(dist)[:10]
            idx = idx[top_idx]

        histories = [history_texts[i] for i in idx]
        persona_histories.append(histories)
        history_string = "\n".join(histories)
        prompt = prompt_profile_template[args.task_name]["profile_prompt"].format(history_string)
        persona_prompts.append(prompt)

    persona_profiles = []
    batches = split_batch(persona_prompts, args.batch_size)

    with torch.no_grad():
        for batch in batches:
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                return_token_type_ids=False,
            ).to(llm.device)

            with torch.autocast(device_type="cuda"):
                outputs = llm.generate(
                    **inputs,
                    do_sample=True,
                    top_k=10,
                    top_p=0.9,
                    temperature=0.6,
                    eos_token_id=tokenizer.eos_token_id,
                    max_new_tokens=300,
                )
            
            outputs = tokenizer.batch_decode(
                outputs,
                skip_special_tokens=True,
            )

            for prompt, output in zip(batch, outputs):
                profile = output.replace(prompt, '').strip()
                persona_profiles.append(profile)
        
    persona_database[user_id] = {
        # "prompts": persona_prompts,
        "profiles": persona_profiles,
        # "centers": centers.tolist(),
    }

print("Saving persona database...")

save_path = (
    f"./data/{args.task_name}/"
    f"persona-profile-k{args.persona_num}-test.json"
)

with open(save_path, "w") as f:
    json.dump(
        persona_database,
        f,
        indent=4,
        ensure_ascii=False
    )

print(f"Persona profiles saved to {save_path}")
# print("Routing Query to Persona...")

# result = []

# for user in tqdm(test_data):
#     user_id = user["user_id"]
#     persona_info = persona_database[user_id]
#     centers = np.array(persona_info["centers"], dtype=np.float32)

#     profiles = persona_info["profiles"]

#     centers = centers / (np.linalg.norm(centers, axis=1, keepdims=True) + 1e-12)

#     for query in user["query"]:
#         question = query["input"]
#         if args.task_name == "citation":
#             article = extract_citation_title(question)
#             option1 = extract_option(question, 1)
#             option2 = extract_option(question, 2)
#             retrieval_query = (
#                 prompt_template[args.task_name]
#                 ["retrieval_query_wokey"]
#                 .format(
#                     article,
#                     option1,
#                     option2,
#                 )
#             )
#         else:
#             article = extract_article(question)

#             retrieval_query = (
#                 prompt_template[args.task_name]
#                 ["retrieval_query_wokey"]
#                 .format(article)
#             )
        
#         query_embedding = embedding_model.encode(
#             retrieval_query,
#             normalize_embeddings=True,
#             convert_to_numpy=True,
#         )

#         scores = np.matmul(centers, query_embedding)
#         persona_id = int(np.argmax(scores))

#         result.append({
#             "prompt": persona_info["prompts"][persona_id],
#             "user_id": user_id,
#             "query_id": query["id"],
#             "profile": profiles[persona_id],
#         })

# with open(f"./data/{args.task_name}/task-profile-k{args.persona_num}-test.json", "w") as f:
#     json.dump(result, f, indent=4)