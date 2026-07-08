import numpy as np
import itertools as it
import csv
from tabulate import tabulate

# Scenario 2: Ring Wear
# KV cache sits at a fixed location in memory and each query writes starting from where the previous query ended

models = {
    "llama3-1B":   {"param": 1,   "N": 16,  "n_head": 64,  "d_head": 16},
    "llama3-8B":   {"param": 8,   "N": 32,  "n_head": 128, "d_head": 32},
    "llama3-70B":  {"param": 70,  "N": 80,  "n_head": 64,  "d_head": 128},
    "llama3-405B": {"param": 405, "N": 126, "n_head": 128, "d_head": 128},
}

memory_GB = [2**i for i in range(10)]
memory_bits = [(2**i) * 8e9 for i in range(10)]

attention_types = ["GQA", "MHA", "MQA"]

b = {
    "INT8/FP8": 8,
    "BF16/FP16": 16,
    "FP32": 32,
}

# W = 10**5   # RRAM
# W = 10**15  # STT-MRAM
W = 10**13    # FeRAM

t_life = 5 * 3.154e7

# Made-up example query lengths (sequence (S) for each entry) in tokens
S = [87, 132, 640, 51, 2100, 18, 67, 99] # Placeholder values until replaced


def get_H_KV(attention_type, n_head):
    if attention_type == "MHA":
        return n_head
    elif attention_type == "MQA":
        return 1
    elif attention_type == "GQA":
        return 8


def get_B_token(N, H_KV, d_head, b):
    return N * (2 * H_KV * d_head) * b


def get_R_max(W, N_bits, B_token, t_life):
    return (W * N_bits) / (B_token * t_life)
