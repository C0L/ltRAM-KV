import numpy as np
import matplotlib.pyplot as plt

# Parameters
write_energy = 400.0 * 1e-12 # J/bit
read_energy  = 1.0 * 1e-12 # J/bit

# llama3-70B, GQA, BF16 
N = 80
H_KV = 8
d_head = 128
b = 16

b_token = N * 2 * H_KV * d_head * b # bits of KV per token, all layers

# For understanding:
# prompt_length = 500 # sequence length basically
# response_length = prompt_length # In this model, the response length is equal to the prompt length

# # prefill: write only, no LtRAM reads
# prefill_write_bits = prompt_length * b_token # technically B_total

# # decode
# KVcache_length = prompt_length # in TOKENS, from the prefill, the KV cache is already filled with prompt_length tokens
# decode_write_bits = 0
# decode_read_bits  = 0

# for i in range(1, response_length + 1):
#     decode_write_bits += b_token # for every token generated
#     decode_read_bits += KVcache_length * b_token
#     KVcache_length += 1

# # Getting total read and write bits for prompt and request tgt
# total_write_bits = prefill_write_bits + decode_write_bits
# total_read_bits  = decode_read_bits
# total_energy = total_write_bits * write_energy + total_read_bits * read_energy # in Joules


prompt_lengths = [2**i for i in range(2, 18)]   # 4, 8, 16, ... 32768
energy_ratios = []
 
for prompt_length in prompt_lengths:

    response_length = prompt_length # response length is equal to the prompt length
    prefill_write_bits = prompt_length * b_token
    KVcache_length = prompt_length
    decode_write_bits = 0
    decode_read_bits  = 0
 
    for i in range(1, response_length + 1):
        decode_write_bits += b_token
        decode_read_bits += KVcache_length * b_token
        KVcache_length += 1
 
    total_write_bits = prefill_write_bits + decode_write_bits
    total_read_bits  = decode_read_bits
 
    energy_ratios.append((total_read_bits * read_energy) / (total_write_bits * write_energy)) # Joules / Joules 
  
 

