"""
Generate profile using:

History
    |
LLM Self-Consistency Scoring
    |
Top-K Representative History Selection
    |
LLM Profile Generation


Compatible with LaMP tasks.
"""


import os
import json
import argparse

import torch

from tqdm import tqdm

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

import random
from scorer import HistoryScorer
from selector import HistorySelector
from generator import ProfileGenerator

from prompts import format_history

# from ./utils import (
#     get_first_k_tokens,
# )


def get_first_k_tokens(text, k):
    """
    Extracts the first k tokens from a text string.

    :param text: The input text string.
    :param k: The number of tokens to extract.
    :return: The first k tokens of the text string.
    """
    # Split the text into tokens based on whitespace
    tokens = text.split()
    output = " ".join(tokens[:k])

    # Return the first k tokens
    return output
##############################################################
# Arguments
##############################################################

def parse_args():

    parser = argparse.ArgumentParser(
        description="Self Consistency Profile Generation"
    )


    parser.add_argument(
        "--model_name",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.2",
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
        "--topk",
        type=int,
        default=10,
    )


    parser.add_argument(
        "--sample_times",
        type=int,
        default=5,
    )


    parser.add_argument(
        "--max_history_tokens",
        type=int,
        default=368,
    )


    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
    )


    parser.add_argument(
        "--output_dir",
        type=str,
        default="./citation",
    )

    parser.add_argument(
        "--ratio",
        type=float,
        default=0.5,
    )

    return parser.parse_args()



##############################################################
# Load Model
##############################################################

def load_llm(model_name):


    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side="left",
    )


    tokenizer.eos_token="</s>"

    tokenizer.pad_token="[PAD]"



    model = AutoModelForCausalLM.from_pretrained(

        model_name,

        device_map="auto",

        trust_remote_code=True,

        torch_dtype=torch.float16,

    )


    model.config.use_cache=True

    model.config.pad_token_id = (
        tokenizer.pad_token_id
    )

    model.config.eos_token_id = (
        tokenizer.eos_token_id
    )


    return tokenizer, model



##############################################################
# Load Data
##############################################################

def load_data(args):


    if args.input_file is not None:

        path=args.input_file

    else:

        path = (
            f"../data/"
            f"{args.task_name}/"
            f"test_100/"
            f"user_test_100.json"
        )


    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        data=json.load(f)


    return data



##############################################################
# Build History Text
##############################################################

def build_history(
    user,
    task_name,
    ratio=0.5,
    max_tokens=368,
):


    history_texts=[]

    profile = user["profile"]
    history_num = len(profile)
    use_num = history_num

    if ratio < 1.0:
        use_num = max(1, int(history_num * ratio))

    if use_num < history_num:
        profile = random.sample(profile, use_num)

    for item in profile:
        item=item.copy()
        for key,value in item.items():
            if isinstance(value,str):
                item[key]=get_first_k_tokens(
                    value,
                    max_tokens
                )
            else:
                item[key]=str(value)



        history=format_history(
            task_name,
            item,
        )


        history_texts.append(
            history
        )


    return history_texts



##############################################################
# Process One User
##############################################################

def process_user(
    args,
    user,
    task_name,
    scorer,
    selector,
    generator,
):


    ##########################################################
    # 1. Build history
    ##########################################################

    histories = build_history(
        user,
        task_name,
        ratio=args.ratio
    )


    if len(histories)==0:

        return {

            "user_id":
            user["user_id"],

            "profile":"",

            "selected_history":[],

        }



    ##########################################################
    # 2. Score history
    ##########################################################

    ranked_histories = (
        scorer.rank_histories(
            histories
        )
    )


    ##########################################################
    # 3. Select Top-K
    ##########################################################

    selected = selector.select(
        ranked_histories
    )



    ##########################################################
    # 4. Generate Profile
    ##########################################################

    profile = generator.generate_profile(
        selected
    )



    ##########################################################
    # 5. Save intermediate information
    ##########################################################

    return {


        "user_id":
        user["user_id"],


        "profile":
        profile,


        "selected_history":
        selected,


        "history_scores":
        ranked_histories,


    }



##############################################################
# Save
##############################################################

def save_json(
    data,
    path,
):

    os.makedirs(
        os.path.dirname(path),
        exist_ok=True
    )


    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )



##############################################################
# Main
##############################################################

def main():


    args=parse_args()



    print("Loading model...")

    tokenizer, model = load_llm(
        args.model_name
    )



    print("Loading data...")

    data=load_data(args)



    ##########################################################
    # Initialize modules
    ##########################################################

    scorer=HistoryScorer(
        tokenizer=tokenizer,
        model=model,
        sample_times=args.sample_times,
    )


    selector=HistorySelector(
        method="topk",
        topk=args.topk,
    )



    generator=ProfileGenerator(

        tokenizer=tokenizer,

        model=model,

        task_name=args.task_name,

    )



    ##########################################################
    # Process
    ##########################################################

    results=[]


    for user in tqdm(data):
        result=process_user(
            args,
            user,
            args.task_name,
            scorer,
            selector,
            generator,
        )


        results.append(
            result
        )

    ##########################################################
    # Save
    ##########################################################

    save_json(
        results,
        os.path.join(
            args.output_dir,
            "profile_results.json"
        )
    )



    ##########################################################
    # Save simplified profile file
    ##########################################################

    profile_dict={}


    for item in results:

        profile_dict[
            item["user_id"]
        ] = item["profile"]



    save_json(profile_dict, os.path.join(args.output_dir,"profile-id2text_test111.json"))
    print("Done!")

if __name__=="__main__":

    main()