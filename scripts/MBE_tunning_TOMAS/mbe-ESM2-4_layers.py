import pandas as pd
import numpy as np
import glob
import os
import torch
import lightning as L
from lightning.pytorch.loggers import WandbLogger
import esm

# import classes
from common_classes import MBEDataset, MBEDataModule, MBEFeedForward, LitMBE

# log into to wandb to log results
import wandb

torch.set_num_threads(24)

WANDB_NOTEBOOK_NAME = '2024-06-13_mbe-linear.ipynb'

wandb.login()

datadir = './out/corrected/counts'
procdir = './out/modelling/processed'
BATCH_SIZE=64


os.makedirs(procdir, exist_ok=True)
np.random.seed(12345)
rerun_encoding = False


# get files to be imported
files = glob.glob(os.path.join(datadir, '2024-06-05_r[0-1]_np-cc*aa-seq-counts.tsv.gz'))

# read tsv files
df = pd.concat([pd.read_csv(f, sep='\t').assign(round = os.path.basename(f)[11:13]) for f in files], axis=0)

# sum counts within each round
df = df.groupby(['round', 'sequence']).sum().reset_index()


pseduo = 1
df_unique = (df
             # add pseudocunt
             .assign(count = lambda x: x['count'] + pseduo)
             # r0 and 1 counts in separate columns
             .pivot(index='sequence', columns='round', values='count')
             .fillna(pseduo)
             .reset_index()
             # calculate log enrichment and weights as per MBE paper
             .assign(
                le = lambda x: np.log2((x['r1']/x['r1'].sum())/(x['r0']/x['r0'].sum())),
                sig = lambda x: 1/x['r1']*(1-x['r1']/(x['r1'].sum())) + 1/x['r0']*(1-x['r0']/(x['r0'].sum())),
                weight = lambda x: 1/(2*x['sig']),
                     )
             )

thresh = 1
df_unique = df_unique.assign(set = lambda x: ['high' if w >= thresh else 'low' for w in x['weight']])

df_test_eval = df_unique.sample(frac=1).groupby('set').head(100)
df_test = df_test_eval.groupby('set').head(50)

# remove test set from evaluation set
df_eval = df_test_eval[~df_test_eval['sequence'].isin(df_test['sequence'])]
df_train = df_unique[~df_unique['sequence'].isin(df_test_eval['sequence'])]

# one hot encoding
aas = [i for i in 'ACDEFGHIKLMNPQRSTVWY*']

def esm_embedding(seqs, max_len):
    # print(seqs)
    
    # data is a list of tuples (index, sequence)
    data = [(str(round_val), sequence + "<pad>" * (max_len - len(sequence))) for round_val, sequence in seqs]
    
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()

    batch_converter = alphabet.get_batch_converter(max_len)
    model.eval()

    batch_size = 2 ** 5 # = 32
    prev = 0
    next = batch_size if batch_size < len(data) else len(data)

    labels = []
    sequence_representations = []
    while next <= len(data) and next != prev:
        # print("Prev: ", prev, " and Next: ", next)
        batch_labels, batch_seq, batch_tokens = batch_converter(data[prev:next])
        labels = labels + batch_labels
        # print("Finished the batch items")
        # batch_lens is just an array with the length of all of the sequences
        # and alphabet.padding_idx is jus thte index of the token '<pad>' in the alpahabet token list
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=True)
        token_representations = results["representations"][33]
        # print("Finished no grad")
        # Generate per-sequence representations via averaging
        # # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
        # sequence_representations = []
        for i, tokens_len in enumerate(batch_lens):
            sequence_representations.append((token_representations[i, 1 : tokens_len - 1].mean(1)).numpy())
        # print('Finished appending sequence representations')
        prev = next
        if next + batch_size > len(data):
            next = len(data)
        else:
            next += batch_size

        del batch_lens
        del batch_tokens
        del batch_seq
        del batch_labels
        del results
        del token_representations

    
    return labels, sequence_representations



def encode(df, max_len):
    # Extract the index (or this case the round number) and the amino acid sequence of each entry in the data frame
    # index_and_sequence = [(round_val, df.loc[round_val, 'sequence'][:len(df.loc[round_val, 'sequence']) - 1]) for round_val in df.index.tolist()]
    index_and_sequence = [(round_val, df.loc[round_val, 'sequence'].replace('*', '<eos>')) for round_val in df.index.tolist()]
    indexes, encoded_numpy_arrr= esm_embedding(index_and_sequence, max_len)
    # print("Finished embedding")
    # numpy_arrays = [tensor.numpy() for tensor in encoded_tensor]
    
    df_encoded = pd.DataFrame({
        'round': indexes,
        'encoded': encoded_numpy_arrr
    })

    df_encoded.set_index('round', inplace=True)
    df_encoded.index = df_encoded.index.astype('int64')

    out = df
    # out['encoded'] = df_encoded
    out.loc[:,'encoded'] = df_encoded
    
    return out


max_seq_length = df_unique['sequence'].str.len().max() + 10
# rerun_encoding = True

if not os.path.exists(os.path.join(procdir, 'train_ESM_embedding.parquet')) or rerun_encoding:
    print("Saving to files")
    
    df_train = encode(df_train, max_seq_length)
    print('Saved Train')
    df_eval = encode(df_eval, max_seq_length)
    print("Saved Eval")
    df_test = encode(df_test, max_seq_length)
    print("Saved Test")
    # save data
    df_train.to_parquet(os.path.join(procdir, 'train_ESM_embedding.parquet'))
    df_eval.to_parquet(os.path.join(procdir, 'eval_ESM_embedding.parquet'))
    df_test.to_parquet(os.path.join(procdir, 'test_ESM_embedding.parquet'))

else:
    #df_train = pd.read_parquet(os.path.join(procdir, 'train_ESM_embedding.parquet')) # too large
    df_eval = pd.read_parquet(os.path.join(procdir, 'eval_ESM_embedding.parquet'))
    df_test = pd.read_parquet(os.path.join(procdir, 'test_ESM_embedding.parquet'))
    # max_seq_length would not be defined if the dfs were already loaded in an external file
    
# test out dataset class
ds = MBEDataset(df_eval)

# test out datamodule class - takes a while to load pickled data
dm = MBEDataModule(procdir, batch_size=BATCH_SIZE)
dm.setup(encode = 'ESM_embedding.parquet')


model = MBEFeedForward(len(ds[0][0]), pos_weight=ds.get_weights(), num_layers=3)

# instantiate lightning model
lit_model = LitMBE(model, pos_weight=ds.get_weights())

# use weights and biases logger
wandb_logger = WandbLogger(project='mbe', name = "feedforward ESM2")
wandb_logger.experiment.config.update({
    "lr": 0.001,
    "pos_weight": ds.get_weights(),
    "n_units": 128,
    "batch_size": BATCH_SIZE,
    "max_seq_length": max_seq_length,
    "arch": "feedforward",
    "enc": "ESM2",
    "loss": "BCEWithLogitsLoss",
    "opt": "Adam",
    "weight_decay": 0.00005,
    "layers" : 4
})

# train model
trainer = L.Trainer(max_epochs=10, logger = wandb_logger)
trainer.validate(model = lit_model, datamodule = dm)
trainer.fit(model = lit_model, datamodule = dm)
wandb.finish()