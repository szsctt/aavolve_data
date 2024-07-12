import os
import glob
import argparse

import numpy as np
import pandas as pd

np.random.seed(12345)
AAs = 'ACDEFGHIKLMNPQRSTVWY*'

def main():

    parser = argparse.ArgumentParser(description='Encode data')
    parser.add_argument('--input', type=str, help='Input data directory', required=True)
    parser.add_argument('--encoding-type', choices=['label', 'onehot'], help='Encoding type', default = 'onehot')
    parser.add_argument('--weight-threshold', type=float, help='Weight threshold', default = 1)
    parser.add_argument('--n-test', type=int, help='Number of test sequences for each weight class', default = 50)
    parser.add_argument('--max-len', type=int, help='Maximum sequence length (if none, 10 + max length in dataset)')
    parser.add_argument('--output', type=str, help='Output data directory')
    args = parser.parse_args()

    # read input files
    df = import_data(args.input)

    # encode
    df = encode(df, args.encoding_type, args.max_len)

    # train/eval/test split
    df_train, df_eval, df_test = split_sequences(df, args.weight_threshold, args.n_test)

    # save
    os.makedirs(args.output, exist_ok=True)
    df_train.to_pickle(os.path.join(args.output, f'train_{args.encoding_type}.pkl.zstd'))
    df_eval.to_pickle(os.path.join(args.output, f'eval_{args.encoding_type}.pkl.zstd'))
    df_test.to_pickle(os.path.join(args.output, f'test_{args.ecndogin_type}.pkl.zstd'))


def encode(df, encoding_type, max_len = None):
    """
    Encode sequences
    """

    if max_len is None:
        max_len = df['sequence'].str.len().max()  + 10

    if encoding_type == 'label':
        df = df.assign(encoded = lambda x: x['sequence'].apply(label_encode, max_len = max_len))

    elif encoding_type == 'onehot':

        df = df.assign(encoded = lambda x: x['sequence'].apply(onehot_encode, max_len = max_len))

    else:
        raise ValueError('Invalid encoding type')
    
    return df

def label_encode(seq, max_len):
    """
    Label encode a sequence
    """
    assert len(seq) <= max_len
    out = np.zeros((max_len, 1))
    
    for i in range(len(seq)):
        out[i] = AAs.index(seq[i]) + 1 # 0 is reserved for padding
    
    return out

def onehot_encode(seq, max_len):
    """
    One-hot encode a sequence
    """

    assert len(seq) <= max_len
    out = np.zeros((max_len, len(AAs)))
    
    for i in range(len(seq)):
        out[i, AAs.index(seq[i])] = 1
    
    return out.flatten()


def split_sequences(df, weight_threshold, n_test):
    """
    Split sequences into train, eval, and test sets, getting n_test sequences from each weight class for eval and test
    """

    # classify sequences based on weight
    df = df.assign(weight_class = lambda x: ['high' if w > weight_threshold else 'low' for w in x['weight']])

    # sample n_test sequences from each weight class for eval and test
    df = df.sample(frac=1)
    df_eval_test = df.groupby('weight_class').head(n_test * 2)
    df_eval = df_eval_test.groupby('weight_class').head(n_test)
    df_test = df_eval_test.groupby('weight_class').tail(n_test)
    df_train = df[~df['sequence'].isin(df_eval_test['sequence'])]

    return df_train, df_eval, df_test

def import_data(input_dir, add_psuedocounts = True):
    """
    Import data from input files and 
    """

    # get files to load
    input_files = glob.glob(os.path.join(input_dir, 'r[0-1]_np-cc*aa-seq-counts.tsv.gz'))

    # assume round is first two characters of filename
    df = pd.concat([pd.read_csv(f, sep='\t').assign(round = os.path.basename(f)[:2]) for f in input_files], axis=0)

    # sum counts for same sequence in same round
    df = df.groupby(['round', 'sequence']).sum().reset_index()

    # add pseudocounts
    ps = 1 if add_psuedocounts else 0
    df = df.assign(count = lambda x: x['count'] + ps)

    # get unique sequences
    df = (df
          .pivot(index='sequence', columns='round', values='count')
          .reset_index()
          .fillna(ps)
          )
    
    # calculate log enrichment and weights
    df = df.assign(
                le = lambda x: np.log2((x['r1']/x['r1'].sum())/(x['r0']/x['r0'].sum())),
                sig = lambda x: 1/x['r1']*(1-x['r1']/(x['r1'].sum())) + 1/x['r0']*(1-x['r0']/(x['r0'].sum())),
                weight = lambda x: 1/(2*x['sig']),
                )

    return df

if __name__ == '__main__':
    main()