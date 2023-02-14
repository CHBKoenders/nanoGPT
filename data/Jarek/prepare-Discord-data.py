import os
import pandas
import tiktoken
import numpy as np
from numpy.random import default_rng

rng = default_rng()

# Access the scraped discord messages
dir = "../DiscordData/Jarek"


def check_time(curr_time, time, curr_half, half, window):
    # get hour minutes in integers
    curr_h, curr_m = curr_time.split(':')
    h, m = time.split(':')
    curr_h = int(curr_h)
    h = int(h)
    curr_m = int(curr_m)
    m = int(m)

    # convert to 24h
    if curr_half == "PM":
        curr_h += 12
    if half == "PM":
        h += 12

    if (h  - curr_h) % 24 < 1:
        return (h - curr_h) % 60 <= window
    
    return False


def check_date_and_time(curr_date, date, window):
    date, time, half = date.split(' ')
    curr_date, curr_time, curr_half = curr_date.split(' ')
    if date == curr_date:
        return check_time(curr_time, time, curr_half, half, window)
    return False

messages = []
for fnamne in os.listdir(dir):
    path = os.path.join(dir, fnamne)
    """
        TODO: combine messages that are close in time into a single message.
    """
    df = pandas.read_csv(path)
    df = df[~df['Content'].isnull()]
    window = 1  # in minutes
    sample = df.iloc[0]
    curr_date = 'x x x'
    curr_msg = ''
    for i in range(len(df)):
        sample = df.iloc[i]
        date = sample['Date']
        msg = sample['Content']
        valid = check_date_and_time(curr_date, date, window)

        if not valid: messages.append(curr_msg)

        curr_msg = curr_msg + '\n' + msg if valid else msg
        curr_date = date if not valid else curr_date
messages = np.array(messages)[1:]
n = len(messages)

# p = rng.permutation(n)
p = np.arange(n)  # not permuting as message order is of importance
# Maybe to reduce bias we could do mini-batch shuffeling or so?
frac = 0.9

train_data = messages[p[:int(n * frac)]]
valid_data = messages[p[int(n * frac):]]
print(f"Splitting data into {len(train_data)} training, and {len(valid_data)} "
      "validation messages")
    
# encode with tiktoken gpt2 bpe
enc = tiktoken.get_encoding("gpt2")
def encode_messages(messages):
    eot_token = enc.eot_token
    ids = [eot_token] # eot seperates messages, so indicates start as well
    for msg in messages:
        msg = "JarekGTP: "+msg # condition on answer-style
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