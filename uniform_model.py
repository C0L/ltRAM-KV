import numpy as np
import itertools as it
import csv
from tabulate import tabulate

# This code is implementing Scenario 1: Uniform Wear

# Nested dictionary of models with their parameters
models = {
    "llama3-1B":   {"param": 1,   "N": 16,  "n_head": 64,  "d_head": 16},
    "llama3-8B":   {"param": 8,   "N": 32,  "n_head": 128, "d_head": 32},
    "llama3-70B":  {"param": 70,  "N": 80,  "n_head": 64,  "d_head": 128},
    "llama3-405B": {"param": 405, "N": 126, "n_head": 128, "d_head": 128},
}

# Memory sizes in GB and bits (table shows GB for readability)
memory_GB = [2**i for i in range(10)]
memory_bits = [(2**i) * 8e9 for i in range(10)]

# Attention types and numerical precision (b) in bits per element
attention_types = ["GQA", "MHA", "MQA"]

b = {
    "INT8/FP8": 8,
    "BF16/FP16": 16,
    "FP32": 32,
}

# Maximum endurance writes (W) and device lifespan in seconds (t_life)
# W = 10**5 # RRAM Endurance 
# W = 10**15 # STT-MRAM Endurance
W = 10**13 # FeRAM Endurance

t_life = 5 * 3.154e7 

# Functions to get H_KV, B_token, and R_max
def get_H_KV(attention_type, n_head):
    if attention_type == "MHA":
        return n_head
    elif attention_type == "MQA":
        return 1
    elif attention_type == "GQA":
        return 8 # GQA KV heads set as 8, should the number change?? 


def get_B_token(N, H_KV, d_head, b):
    return N * (2 * H_KV * d_head) * b


def get_R_max(W, N_bits, B_token, t_life):
    return (W * N_bits) / (B_token * t_life)


entries = []
for precision, bit in b.items():
    for attention_type, (model_name, model) in it.product(attention_types, models.items()):

        # Extract model parameters
        param = model["param"]
        N = model["N"]
        n_head = model["n_head"]
        d_head = model["d_head"]
        H_KV = get_H_KV(attention_type, n_head)
        B_token = get_B_token(N, H_KV, d_head, bit) # bits/token

        row = [model_name, precision, attention_type, H_KV, B_token]

        for N_GB, N_bits in zip(memory_GB, memory_bits):

            if N_GB < param:
                row.append(np.nan) # Originally "NA"
                continue

            R_max = get_R_max(W, N_bits, B_token, t_life)

            row.append(round(R_max))

        entries.append(row)


headers = (
    ["Model", "Precision", "Attention", "H_KV", "B_token"] + [f"{GB} GB" for GB in memory_GB]
)

print(tabulate(entries, headers=headers))

# Below code saves the results as a .csv file for potential data analysis abd visualizations
with open("CHANGE_NAME_OR_DELETE.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(headers)
    writer.writerows(entries)

print("Results are saved as .csv file!")

# CHANGE_NAME_OR_DELETE