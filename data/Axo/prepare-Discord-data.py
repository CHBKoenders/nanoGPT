import os
import argparse
import pandas
import tiktoken
import numpy as np
from numpy.random import default_rng

rng = default_rng()
# encode with tiktoken gpt2 bpe
enc = tiktoken.get_encoding("gpt2")
EOT_TOKEN = enc.eot_token

def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--in-dir', type=str, default='./Export')
    argparser.add_argument('--out-dir', type=str, default='.')
    return argparser.parse_args()

def process_msgs(contents, authors):
    ids = []
    for msg, user in zip(contents, authors):
        msg = user+":\n"+msg+"\n"
        curr_ids = enc.encode_ordinary(msg)
        curr_ids.append(EOT_TOKEN)
        ids += curr_ids
    return ids

def process_df(df, train_data, valid_data):
    df = df[df['Content'].notnull()]
    contents = df['Content']
    authors = df['Author']
    n = len(df)
    n_train = int(0.9*n)

    train_data += process_msgs(contents.iloc[:n_train], authors.iloc[:n_train])
    valid_data += process_msgs(contents.iloc[n_train:], authors.iloc[n_train:])

    return train_data, valid_data

if __name__ == '__main__':
    args = parse_args()

    train_data, valid_data = [], []
    for fname in os.listdir(args.in_dir):
        if fname.split('.')[-1] != "csv": continue
        path = os.path.join(args.in_dir, fname)
        df = pandas.read_csv(path)
        train_data, valid_data = process_df(df, train_data, valid_data)


    print(f"train has {len(train_data):,} tokens")
    print(f"val has {len(valid_data):,} tokens")

    # export to bin files
    train_ids = np.array(train_data, dtype=np.uint16)
    val_ids = np.array(valid_data, dtype=np.uint16)
    train_ids.tofile(os.path.join(args.out_dir, 'train.bin'))
    val_ids.tofile(os.path.join(args.out_dir, 'val.bin'))