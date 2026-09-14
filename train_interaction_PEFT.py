import os
import json
import argparse
import copy
import datetime
from tqdm import tqdm
from pathlib import Path
from functools import partial
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset
from llama import Tokenizer
from engine_finetuning import (
    train_one_epoch,
    load_model
)
from rank_bm25 import BM25Okapi
from utils import (
    get_first_k_tokens,
)

# =====================
# Dataset
# =====================
class InstructionDataset(Dataset):
    def __init__(self, data_list, tokenizer_path, max_tokens=2048):
        self.data_list = data_list
        self.tokenizer = Tokenizer(model_path=tokenizer_path+"/tokenizer.model")
        self.max_tokens = max_tokens

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        prompt = torch.tensor(
            self.tokenizer.encode(
                item["prompt"],
                bos=True,
                eos=False
            ),
            dtype=torch.int64
        )

        example = torch.tensor(
            self.tokenizer.encode(
                item["full_example"],
                bos=True,
                eos=True
            ),
            dtype=torch.int64
        )

        labels = copy.deepcopy(example)
        labels[:len(prompt)] = -1

        return example, labels, prompt
    
def collate_fn(batch, max_length=2048):
    examples, labels, prompts = zip(*batch)
    # Trim sequences to max_length
    trimmed_examples = [example[:max_length] for example in examples]
    trimmed_labels = [label[:max_length] for label in labels]
    
    # Determine the maximum sequence length after trimming but capped at max_length
    max_length = min(max([len(example) for example in trimmed_examples]), max_length)

    # Pad sequences to the determined max_length
    padded_examples = torch.stack([torch.cat((example, torch.zeros(max_length - len(example), dtype=torch.int64) - 1)) if len(example) < max_length else example for example in trimmed_examples])
    padded_labels = torch.stack([torch.cat((label, torch.zeros(max_length - len(label), dtype=torch.int64) - 1)) if len(label) < max_length else label for label in trimmed_labels])

    example_masks = padded_examples.ge(0)
    label_masks = padded_labels.ge(0)

    padded_examples[~example_masks] = 0
    padded_labels[~label_masks] = 0

    example_masks = example_masks.float()
    label_masks = label_masks.float()

    return padded_examples, padded_labels, example_masks

parser = argparse.ArgumentParser("MAE pre-training", add_help=False)
parser.add_argument(
    "--batch_size",
    default=1,
    type=int,
    help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus",
)
parser.add_argument("--epochs", default=3, type=int)
parser.add_argument("--max_step", default=1000, type=int)

parser.add_argument("--warmup_epochs", default=0, type=int)

parser.add_argument(
    "--accum_iter",
    default=1,
    type=int,
    help="Accumulate gradient iterations (for increasing the effective batch size under memory constraints)",
)

# Model parameters
parser.add_argument("--llama_model_path", default="/disk/eyzhang/.cache/modelscope/hub/models/shakechen/Llama-2-7b", type=str, help="path of llama model")
parser.add_argument("--tokenizer_path", default="/disk/eyzhang/.cache/modelscope/hub/models/shakechen/Llama-2-7b", type=str, help="path of llama model tokenizer")
parser.add_argument("--group_content_path", default="./anchor_selection/news_headline/all_interaction/clusters")
parser.add_argument("--task_name", default="news_headline", type=str, metavar="MODEL", help="name of the task")

parser.add_argument("--model", default="llama7B_lora", type=str, metavar="MODEL", help="Name of model to train")
parser.add_argument("--max_seq_len", type=int, default=3000, metavar="LENGTH", help="the maximum sequence length")

parser.add_argument("--w_lora", type=bool, default=True, help="use lora or not")

# Optimizer parameters
parser.add_argument("--weight_decay", type=float, default=0.01, help="weight decay (default: 0.05)")

parser.add_argument("--lr", type=float, default=2e-5, help="learning rate (absolute lr)")
parser.add_argument("--clip", type=float, default=0.3, help="gradient clipping")

parser.add_argument(
    "--blr",
    type=float,
    default=1e-3,
    metavar="LR",
    help="base learning rate: absolute_lr = base_lr * total_batch_size / 256",
)
parser.add_argument(
    "--min_lr", type=float, default=0.0, metavar="LR", help="lower lr bound for cyclic schedulers that hit 0"
)


# Dataset parameters
# parser.add_argument("--test_data_path", default="./data/news_headline/user_anchor_candidate.json", type=str, help="dataset path")
# parser.add_argument("--train_data_path", default="/afs/crc.nd.edu/user/z/ztan3/Private/LoRA-composition/LaMP_data-final/movie/user_base_LLM.json", type=str, help="dataset path")

parser.add_argument("--output_dir", default="./output/news_headline/all_interaction_cluster", help="path where to save, empty for no saving")

parser.add_argument("--log_dir", default="./output", help="path where to tensorboard log")
parser.add_argument("--device", default="cuda", help="device to use for training / testing")
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--lora_ckpt", default='./output/news_headline/task-base_LLM/lora_ckpt.pt', help="resume lora from checkpoint")
parser.add_argument("--grad_ckpt", type=bool, default=True, help="whether to user gradient checkpoint, recommend TRUE!!")

parser.add_argument("--anchor_path", default='./anchor_selection/news_headline/anchor_user_idx.pt', help="resume lora from checkpoint")

parser.add_argument("--num_workers", default=10, type=int)

parser.add_argument(
    "--pin_mem",
    action="store_true",
    help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
)

parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
parser.set_defaults(pin_mem=True)

# generation hyperparameters
parser.add_argument("--top_p", type=float, default=0.9, help="top_p")
parser.add_argument("--temperature", type=float, default=0.1, help="temperature")
parser.add_argument("--max_gen_len", type=int, default=100, help="top_p")

parser.add_argument("--k_list", type=list, default=[1,2,4], help="top_p")
parser.add_argument('--infer', default=False, action=argparse.BooleanOptionalAction)
args=parser.parse_args()

with open("./prompt/prompt.json", "r", encoding="utf-8") as f:
    prompt_template = json.load(f)

with open(f"./data/{args.task_name}/user_anchor_candidate.json", "r", encoding="utf-8") as f:
    anchor_candidate = json.load(f)

all_interactions = []
for user_idx, user in enumerate(tqdm(anchor_candidate)):
    for history_idx, p in enumerate(user['profile']):
        all_interactions.append(p)


def process_train_data_for_citation(history):
    train_data = []
    profile = history
    instruction_template = "### User History:\n{history}\n\n### User Instruction:\n{query_input}"

    for i, target_item in enumerate(tqdm(profile, desc="Processing training data")):
        history_candidates = profile[:i] + profile[i+1:]
        target_title = target_item.get("title", "")
        target_citation = target_item.get("citation", "")

        distractor_candidates = [item for item in history_candidates if item.get('citation') != target_citation]

        distractor_titles = [item.get("title", "") for item in distractor_candidates]
        tokenized_corpus = [title.split() for title in distractor_titles]
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = target_title.split()
        top_indices = bm25.get_top_n(tokenized_query, list(range(len(distractor_titles))), n=1)

        distractor_item = distractor_candidates[top_indices[0]] if top_indices else None
        distractor_citation = distractor_item.get("citation", "") if distractor_item else ""

        if random.random() < 0.5:
            option1, option2, correct_answer = target_citation, distractor_citation, "[1]"
        else:
            option1, option2, correct_answer = distractor_citation, target_citation, "[2]"

        history_texts = [f"Paper Title: {item.get('title', '')}, Citation: {item.get('citation', '')}" for item in history_candidates[:1]]
        history_str = "\n".join(history_texts)

        query_input = f"For an author who has written the paper with the title \"{target_title}\", which reference is related? Just answer with [1] or [2] without explanation.\nReference: [1] - {option1} [2] - {option2}\nAnswer:"
        
        instruction = instruction_template.format(history=history_str, query_input=query_input)

        prompt = instruction
        full_prompt = instruction + correct_answer
        train_data.append(
        {
            "prompt": prompt,
            "full_example": full_prompt
        })

    return train_data

def process_cluster_data(history):
    data = []
    for q in history:
        # print(q)
        # q = h["data"]
        for k, v in q.items():
            if isinstance(v, str):
                q[k] = get_first_k_tokens(v, 768)
            else:
                q[k] = str(v)

        prompt = prompt_template[args.task_name]["OPPU_input"].format(**q)
        full_prompt = prompt_template[args.task_name]["OPPU_full"].format(**q)

        data.append(
            {
                "prompt": prompt,
                "full_example": full_prompt
            }
        )

    return data

# =====================
# Main
# =====================
def main():
    torch.set_default_device("cuda")
    device = "cuda"

    model = load_model(
        ckpt_dir=args.llama_model_path,
        tokenizer_path=args.tokenizer_path,
        max_seq_len=args.max_seq_len,
        max_batch_size=args.batch_size,
        lora_path=args.lora_ckpt,
        w_lora=True,
        grad_ckpt=True
    )

    model.to(device)
    model.merge_lora_parameters()
    print("Base model loaded")

    output_root=Path(args.output_dir)

    output_root.mkdir(parents=True, exist_ok=True)
    cluster_files = list(Path(args.group_content_path).glob("cluster_*.json"))
    # print(cluster_files)

    for cluster_file in tqdm(cluster_files):
        with open(cluster_file, "r", encoding="utf-8") as f:
            cluster = json.load(f)

        interaction_indices = cluster["interaction_indices"]
        history = [all_interactions[i] for i in interaction_indices]
    
        print(
            "Traing",
            # group_id,
            cluster_file.name,
            "samples:",
            len(history)
        )

        model.reset_lora_parameters()
        model.set_lora_trainable()
        
        if args.task_name == "citation":
            train_data = process_train_data_for_citation(history)
        else:
            train_data = process_cluster_data(history)


        print(f"train_data sample: {train_data[0]}")
        dataset = InstructionDataset(
            train_data,
            args.tokenizer_path,
            args.max_seq_len
        )

        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            generator=torch.Generator(device='cuda'),
            collate_fn=partial(collate_fn, max_length=args.max_seq_len)
        )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=0.01
        )

        args.cur_step = 0

        while args.cur_step < args.max_step:
            stats = train_one_epoch(
                model,
                loader,
                optimizer,
                device,
                0,
                None,
                log_writer=None,
                args=args,
            )

            # args.cur_step += 1

        save_name = (cluster_file.stem + ".pt")

        torch.save(model.lora_state_dict(), output_root/save_name)

        print(f"saved:{save_name}")

if __name__=="__main__":
    main()
