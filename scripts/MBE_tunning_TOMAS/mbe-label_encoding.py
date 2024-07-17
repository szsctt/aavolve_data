import pandas as pd
import numpy as np
import glob
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.nn import functional as F
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from scipy import stats
from sklearn.preprocessing import LabelEncoder

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
def label_encoding(seq, max_l):
    """
    Encode a sequence of amino acids with integer valus for each type fo amino acid
    """
    out = np.zeros(max_l)
    label_encoder = LabelEncoder().fit(aas)
    encoded = label_encoder.transform(list(seq))
    out[:len(seq)] = encoded
    return out


# encode datasets
def encode(df, max_seq_length):
    df = (df
          .assign(encoded = df['sequence'].apply(lambda x: label_encoding(x, max_seq_length)))
          .drop('sequence', axis=1)
          )
    return df

max_seq_length = df_unique['sequence'].str.len().max() + 10

if not os.path.exists(os.path.join(procdir, 'train_label_encoding.pkl')) or rerun_encoding:
    # max_seq_length = df_unique['sequence'].str.len().max() + 10
    df_train = encode(df_train, max_seq_length)
    df_eval = encode(df_eval, max_seq_length)
    df_test = encode(df_test, max_seq_length)
    
    # save data
    df_train.to_pickle(os.path.join(procdir, 'train_label_encoding.pkl'))
    df_eval.to_pickle(os.path.join(procdir, 'eval_label_encoding.pkl'))
    df_test.to_pickle(os.path.join(procdir, 'test_label_encoding.pkl'))

else:
    
    #df_train = pd.read_pickle(os.path.join(procdir, 'train.pkl')) # too large
    df_eval = pd.read_pickle(os.path.join(procdir, 'eval_label_encoding.pkl'))
    df_test = pd.read_pickle(os.path.join(procdir, 'test_label_encoding.pkl'))

# test out dataset class
ds = MBEDataset(df_eval)

# test out datamodule class - takes a while to load pickled data
dm = MBEDataModule(procdir, batch_size=BATCH_SIZE)
dm.setup( encode = 'ESM_embedding.pkl')

# TODO: write bash script to which you will pass the variables. One of which will be the number of layers (for this file it's 2...?)

model = MBEFeedForward(len(ds[0][0]), pos_weight=ds.get_weights(), num_layers=2)

# instantiate lightning model
lit_model = LitMBE(model, pos_weight=ds.get_weights())

# use weights and biases logger
wandb_logger = WandbLogger(project='mbe', name = "feedforward label encoding 3 Layers")
wandb_logger.experiment.config.update({
    "lr": 0.001,
    "pos_weight": ds.get_weights(),
    "n_units": 128,
    "batch_size": BATCH_SIZE,
    "max_seq_length": max_seq_length,
    "arch": "feedforward",
    "enc": "label encoding",
    "loss": "BCEWithLogitsLoss",
    "opt": "Adam",
    "weight_decay": 0.00005,
    "layers" : 3
})

# train model
trainer = L.Trainer(max_epochs=10, logger = wandb_logger)
trainer.validate(model = lit_model, datamodule = dm)
trainer.fit(model = lit_model, datamodule = dm)
wandb.finish()