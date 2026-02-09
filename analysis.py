import numpy as np
import itertools as it
from tabulate import tabulate

# param, layers, n_heads, d_head (bytes)
Ms = {
    'llama3-1B'  :[1, 16,  64, 16],
    'llama3-8B'  :[8, 32,  128, 32],
    'llama3-70B' :[70, 80,  128, 64],
    'llama3-405B':[405, 126, 128, 128],
}

# LtRAM memory sizes (GB)
Ls = [2**l for l in range(10)]

# Grouped, Multi-Head, Multi-Query
Ts = ['GQA', 'MHA', 'MQA']
Qs = [1] # btyes

endurance = 10**6 # writes
lifespan = 5      # years

entries = []
# np.zeros((len(models), len(mems), len(quants)))
for T, Q in it.product(Ts, Qs):
    for (name, model) in Ms.items():
        tps = []
        for L in Ls:
            params, n_layers, n_heads, d_head = model
            if L < params:
                tps.append('NA')
                continue
            kv_heads = (8 if 'GQA' in T else 1 if 'MQA' in T else n_heads)
            writable = endurance * L # GB
            pertoken = 2 * n_layers * kv_heads * d_head * Q * 1e-9 # GB
            cycles = writable / pertoken
            tps.append(round(cycles / (lifespan * 3.154e7))) # seconds
        entries.append([name, Q, T] + tps)
    
headers = ['Model', 'Quant', 'Atten.'] + [str(L) for L in Ls]
print(tabulate(entries, headers=headers))
