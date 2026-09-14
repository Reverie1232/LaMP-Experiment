import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer


DATA_PATH = ""

TASK_NAME = "movie_tagging"

BGE_MODEL = "BAAI/bge-small-en-v1.5"

DEVICE = "cuda"

EPOCHS = 20

BATCH_SIZE = 32

LR = 1e-3

TEMPERATURE = 0.07

SAVE_PATH = "user_encoder.pt"

def get_first_k_tokens(text, k):
    tokens = text.split()
    return " ".join(tokens[:k])

with open('./prompt/prompt.json', 'r') as f:
    prompt_template = json.load(f)
