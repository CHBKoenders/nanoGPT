import os
import pandas
import tiktoken
import numpy as np
from numpy.random import default_rng

rng = default_rng()

# Access the scraped discord messages
dir = "./DiscordData/Jarek"
tot = 0

messages = []
for fnamne in os.listdir(dir):
    path = os.path.join(dir, fnamne)
    df = pandas.read_csv(path)
    messages.append(df['Content'])

messages = pandas.concat(messages)
n_tot = len(messages)
# filter out NaN
messages = messages[~messages.isnull()]

n = len(messages)
p = rng.permutation(n)
frac = 0.9

print(f"From a total of {n_tot} messages, {n_tot - n} have been filtered out\n")

train_data = messages.iloc[p[:int(n * frac)]]
valid_data = messages.iloc[p[int(n * frac):]]
print(f"Splitting data into {len(train_data)} training, and {len(valid_data)} "
      "validation messages")
    
# encode with tiktoken gpt2 bpe
enc = tiktoken.get_encoding("gpt2")
eot_token = enc.eot_token
def encode_messages(messages):
    ids = []
    for msg in messages:
        msg = "\n"+msg
        curr_ids = enc.encode_ordinary(msg)
        curr_ids.append(eot_token)
        ids += curr_ids
    return ids
    
train_ids = encode_messages(train_data)
valid_ids = encode_messages(valid_data)

print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(valid_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(valid_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(dir, 'train.bin'))
val_ids.tofile(os.path.join(dir, 'val.bin'))