import pandas as pd
import numpy as np
import glob
import os
import lightning as L
from lightning.pytorch.loggers import WandbLogger

# import classes
from common_classes import MBEDataset, MBEDataModule, MBEFeedForward, LitMBE

# log into to wandb to log results
import wandb


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
def one_hot(seq, max_l):
    """
    One hot encode a sequence of amino acids
    """
    out = np.zeros((max_l, len(aas)))
    for i in range(max_l):
        if i < len(seq):
            out[i, aas.index(seq[i])] = 1
    return out.flatten()


# encode datasets
def encode(df, max_seq_length):
    df = (df
          .assign(encoded = df['sequence'].apply(lambda x: one_hot(x, max_seq_length)))
          .drop('sequence', axis=1)
          )
    return df

max_seq_length = df_unique['sequence'].str.len().max() + 10

if not os.path.exists(os.path.join(procdir, 'train_onehot.pkl')) or rerun_encoding:
    # max_seq_length = df_unique['sequence'].str.len().max() + 10
    df_train = encode(df_train, max_seq_length)
    df_eval = encode(df_eval, max_seq_length)
    df_test = encode(df_test, max_seq_length)
    
    # save data
    df_train.to_pickle(os.path.join(procdir, 'train_onehot.pkl'))
    df_eval.to_pickle(os.path.join(procdir, 'eval_onehot.pkl'))
    df_test.to_pickle(os.path.join(procdir, 'test_onehot.pkl'))

else:
    
    #df_train = pd.read_pickle(os.path.join(procdir, 'train.pkl')) # too large
    df_eval = pd.read_pickle(os.path.join(procdir, 'eval_onehot.pkl'))
    df_test = pd.read_pickle(os.path.join(procdir, 'test_onehot.pkl'))
    
# test out dataset class
ds = MBEDataset(df_eval)

# test out datamodule class - takes a while to load pickled data
dm = MBEDataModule(procdir, batch_size=BATCH_SIZE)
dm.setup()


model = MBEFeedForward(len(ds[0][0]), pos_weight=ds.get_weights())

# instantiate lightning model
lit_model = LitMBE(model, pos_weight=ds.get_weights())

# use weights and biases logger
wandb_logger = WandbLogger(project='mbe', name = "feedforward")
wandb_logger.experiment.config.update({
    "lr": 0.001,
    "pos_weight": ds.get_weights(),
    "n_units": 128,
    "batch_size": BATCH_SIZE,
    "max_seq_length": max_seq_length,
    "arch": "feedforward",
    "enc": "onehot",
    "loss": "BCEWithLogitsLoss",
    "opt": "Adam"
})

# train model
trainer = L.Trainer(max_epochs=10, logger = wandb_logger)
trainer.validate(model = lit_model, datamodule = dm)
trainer.fit(model = lit_model, datamodule = dm)
wandb.finish()