import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
from rank_bm25 import BM25Okapi
import json
from tqdm import tqdm
from utils import extract_citation_title, extract_option, extract_movie, extract_news_cat, extract_news_headline, extract_product_review, extract_scholarly_title, extract_tweet_paraphrasing

from utils import (
    split_batch,
    get_first_k_tokens,
)

parser = argparse.ArgumentParser(description="Generate Query-specific Profile")
parser.add_argument(
    "--model_name",
    type=str,
    default="mistralai/Mistral-7B-Instruct-v0.2",
)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--k", type=int, default=10)
parser.add_argument("--cut_off", type=int, default=2048)
parser.add_argument("--task_name", type=str, default="movie_tagging")

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

##############################################
# Model
##############################################

tokenizer = AutoTokenizer.from_pretrained(
    args.model_name,
    padding_side="left"
)

tokenizer.eos_token = "</s>"
tokenizer.pad_token = "[PAD]"

base_model = AutoModelForCausalLM.from_pretrained(
    args.model_name,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True,
)

base_model.config.use_cache = True
base_model.config.pad_token_id = tokenizer.pad_token_id
base_model.config.eos_token_id = tokenizer.eos_token_id
base_model.config.bos_token_id = tokenizer.bos_token_id

##############################################
# Data
##############################################

# with open(f"./data/{args.task_name}/user_base_LLM.json") as f:
#     train_data = json.load(f)

with open(f"./data/{args.task_name}/test_100/user_test_100.json") as f:
    test_data = json.load(f)

# with open(f"./data/{args.task_name}/user_anchor_candidate.json") as f:
#     anchor_data = json.load(f)

# with open(f"./data/{args.task_name}/user_reserve_10_percent.json") as f:
#     reserve_data = json.load(f)

data = test_data

with open("./prompt/prompt_profile.json") as f:
    prompt_profile_template = json.load(f)

with open("./prompt/prompt.json") as f:
    prompt_template = json.load(f)

##############################################
# Build prompts
##############################################

prompt_list = []
meta_list = []

for user in tqdm(data):

    ##########################################
    # build corpus once
    ##########################################

    visible_history_list = user['profile']

    for p in visible_history_list:
        for key, value in p.items():
            if isinstance(value, str):
                p[key] = get_first_k_tokens(value, 368)
            else:
                p[key] = str(value)

    history_list = [
        prompt_template[args.task_name]["retrieval_history"].format(**p)
        for p in visible_history_list
    ]

    tokenized_corpus = [doc.split() for doc in history_list]
    bm25 = BM25Okapi(tokenized_corpus)

    ##########################################
    # retrieve for every query
    ##########################################

    for qid, query in enumerate(user["query"]):

        q = query.copy()
        test_question = q["input"]

        if args.task_name == "citation":
            test_article = extract_citation_title(test_question)
            option1 = extract_option(test_question, 1)
            option2 = extract_option(test_question, 2)

            tokenized_query = (
                prompt_template[args.task_name]
                ["retrieval_query_wokey"]
                .format(test_article, option1, option2)
                .split()
            )
        else:
            test_article = extract_article(test_question)

            tokenized_query = (
                prompt_template[args.task_name]
                ["retrieval_query_wokey"]
                .format(test_article)
                .split()
            )

        retrieved_history = bm25.get_top_n(
            tokenized_query,
            history_list,
            n=args.k,
        )

        history_string = "\n".join(retrieved_history)

        prompt = prompt_profile_template[args.task_name]["profile_prompt"].format(history_string)

        prompt_list.append(prompt)

        # print(f"User {user['user_id']} Query {qid} Prompt: {prompt}")
        meta_list.append(
            {
                "user_id": user["user_id"],
                "query_id": q["id"],
            }
        )

##############################################
# Inference
##############################################

batched_prompt = split_batch(prompt_list, args.batch_size)

outputs_all = []

with torch.no_grad():

    for batch in tqdm(batched_prompt):

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            return_token_type_ids=False,
        )

        inputs = inputs.to(base_model.device)

        with torch.autocast(device_type="cuda"):

            outputs = base_model.generate(
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

        outputs_all.extend(outputs)

##############################################
# Save
##############################################

result = []

for prompt, output, meta in zip(
    prompt_list,
    outputs_all,
    meta_list,
):

    profile = output.replace(prompt, "").strip()

    result.append(
        {
            "user_id": meta["user_id"],
            "query_id": meta["query_id"],
            "profile": profile,
        }
    )

with open(
    f"./data/{args.task_name}/query_profile.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        result,
        f,
        ensure_ascii=False,
        indent=2,
    )

print(f"Generate {len(result)} query-specific profiles.")