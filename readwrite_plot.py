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

prompt_lengths = [2**i for i in range(2, 19)]   
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
  
# Plot

plt.plot(prompt_lengths, energy_ratios, color = "r", lw = 2, label = "RRAM")
plt.axhline(1.0, color = "k", lw = 1.5, ls = "--")

plt.text(prompt_lengths[0], 1.5, "read-dominated", fontsize = 10, color = "gray")
plt.text(prompt_lengths[0], 0.5, "write-dominated", fontsize = 10, color = "gray")

plt.xscale("log")
plt.yscale("log")
plt.xlabel("Prompt Length (Tokens)")
plt.ylabel("KV Read/Write Energy Ratio")
plt.title("RRAM KV-Energy Asymmetry")

plt.grid(True, which = "both", alpha = 0.3)
plt.legend()
plt.show()

