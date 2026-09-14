import os
import json
import argparse
import copy
import datetime
import random
from pathlib import Path
from functools import partial

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset
from llama import Tokenizer
from engine_finetuning import (
    train_one_epoch,
    load_model
)

from utils import (
    get_first_k_tokens,
)

from rank_bm25 import BM25Okapi
from tqdm import tqdm
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
parser.add_argument("--group_content_path", default="./anchor_selection/movie_tagging/group_content")
parser.add_argument("--task_name", default="movie_tagging", type=str, metavar="MODEL", help="name of the task")

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


parser.add_argument("--output_dir", default="./output/movie_tagging/group_content_only_lora", help="path where to save, empty for no saving")

parser.add_argument("--log_dir", default="./output", help="path where to tensorboard log")
parser.add_argument("--device", default="cuda", help="device to use for training / testing")
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--lora_ckpt", default=None, help="resume lora from checkpoint")
parser.add_argument("--grad_ckpt", type=bool, default=True, help="whether to user gradient checkpoint, recommend TRUE!!")

parser.add_argument("--anchor_path", default='./anchor_selection/movie_tagging/anchor_user_idx.pt', help="resume lora from checkpoint")

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

def build_bm25_from_profile(profile_items: List[Dict]) -> BM25Okapi:
    """Build BM25 index from profile items."""
    corpus = [f"{item.get('description', '')} {item.get('tag', '')}".strip() for item in profile_items]
    tokenized_corpus = [doc.split() for doc in corpus]
    return BM25Okapi(tokenized_corpus)


def process_train_data_for_movie_tagging(history):
    train_data = []
    profile = [item["data"] for item in history]
    instruction_tpl = (
        # "### User History:\n"
        # "Description: {hist_description} Tag: {hist_tag}\n\n"
        "### User Instruction:\n"
        "Which tag does this movie relate to among the following tags? Just answer with the tag name without further explanation. tags: [sci-fi, based on a book, comedy, action, twist ending, dystopia, dark comedy, classic, psychology, fantasy, romance, thought-provoking, social commentary, violence, true story]\n"
        "Description: {query_description} Tag:"
    )

    for i, current_item in enumerate(tqdm(profile, desc="Processing training data")):
        query_description = current_item.get("description", "")
        correct_tag = current_item.get("tag", "")
        # history_pool = profile[:i] + profile[i+1:]
        # bm25 = build_bm25_from_profile(history_pool)
        # query_tokens = query_description.split()
        # top_indices = bm25.get_top_n(query_tokens, list(range(len(history_pool))), n=1)

        # history_item = history_pool[top_indices[0]] if top_indices else random.choice(history_pool)
        # hist_description = history_item.get("description", "N/A")
        # hist_tag = history_item.get("tag", "N/A")

        instruction = instruction_tpl.format(
            # hist_description=hist_description,
            # hist_tag=hist_tag,
            query_description=query_description
        )

        prompt = instruction
        full_prompt = instruction + correct_tag
        train_data.append(
        {
            "prompt": prompt,
            "full_example": full_prompt
        })

    return train_data

def process_cluster_data(history):
    data = []
    for h in history:
        q = h["data"]
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
    group_dirs=list(Path(args.group_content_path).glob("group_*"))

    for group_dir in group_dirs:
        group_id = group_dir.name
        cluster_files=list(
            group_dir.glob(
                "content_cluster_*.json"
            )
        )

        for cluster_file in cluster_files:
            with open(cluster_file) as f:
                cluster = json.load(f)

            history = cluster["history"]
            if len(history) < 5:
                continue

            print(
                "Traing",
                group_id,
                cluster_file.name,
                "samples:",
                len(history)
            )

            model.reset_lora_parameters()
            model.set_lora_trainable()

            train_data = process_train_data_for_movie_tagging(
                history
            )

            print(train_data[0])
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

            save_name = (group_id + "_" + cluster_file.stem + ".pt")

            torch.save(model.lora_state_dict(), output_root/save_name)

            print(f"saved:{save_name}")

if __name__=="__main__":
    main()
