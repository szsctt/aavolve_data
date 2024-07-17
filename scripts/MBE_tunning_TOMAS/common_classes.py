import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import lightning as L
import os
from scipy import stats

"""
Classes include:
MBEDataset
LEDataset
MBEDataModule (Lightning Module)
MBEFeedForward
MBELogisticRegression
LitMBE
"""


class MBEDataset(Dataset):

    def __init__(self, df):
        self.df_raw = df
        self.df = self.create_mbe_dataset(df)
    
    def create_mbe_dataset(self, df):

        df = (df
              .loc[:, ['encoded', 'r0', 'r1']]
              .melt(id_vars=['encoded'], value_vars=['r0', 'r1'], var_name='round', value_name='value')
              .assign(label = lambda x: (x['round'] == 'r1').astype(int))
        )
        df = df.reindex(df.index.repeat(df['value'])).reset_index()

        return df

    def get_weights(self):
        """
        Number of examples in each class
        """
        counts = (self.df
                .groupby('label')
                .size()
               )
        return torch.tensor(counts.loc[1] / counts.loc[0], dtype=torch.float64)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return row['encoded'], float(row['label'])
    
class LEDataset(Dataset):
    """
    Log enrichment dataset
    """

    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):

        row = self.df.iloc[idx]
        return row['encoded'], row['le']
    
class MBEDataModule(L.LightningDataModule):

    def __init__(self, data_dir, batch_size = 32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, encode="onehot.pkl", stage=None):
        self.train = pd.read_pickle(os.path.join(self.data_dir, f'train_{encode}'))
        self.eval = pd.read_pickle(os.path.join(self.data_dir, f'eval_{encode}'))
        self.test = pd.read_pickle(os.path.join(self.data_dir, f'test_{encode}'))

    def train_dataloader(self):
        return DataLoader(MBEDataset(self.train), batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(LEDataset(self.eval), batch_size=self.batch_size)
    
    def test_dataloader(self):
        return DataLoader(LEDataset(self.test), batch_size=self.batch_size)
    
class MBELogisticRegression(nn.Module):
    """
    Class for logistic regression. 
    Returns logits by default, or probabilities if probs=True.
    """

    def __init__(self, input_size, pos_weight=None):
        super().__init__()
        self.linear = nn.Linear(input_size, 1, dtype = torch.float64)
        self.pos_weight = 1 if pos_weight is None else pos_weight

    def forward(self, x, probs=False):
        """
        Return logits or probabilities
        
        """

        logits = self.linear(x)
        if probs:
            return torch.sigmoid(logits)
        else:
            return logits

    def predict(self, x):
        """
        Return density ratio, an estimate of log enrichment
        """
        
        # get probability for positive class
        p = self.forward(x, probs=True)

        # density ratio is p/(1-p), adjusted for class imbalance
        return p/(1-p) / self.pos_weight

def linear_layers(input_size, n_units, num_layers = 1):

    layers = []
    
    # Add the input layer and first ReLU activation
    layers.append(nn.Linear(input_size, n_units, dtype=torch.float64))
    
    # Add the hidden layers
    for x in range(1, num_layers):
        layers.append(nn.ReLU())
        layers.append(nn.Linear(n_units, n_units, dtype=torch.float64))
    
    return nn.Sequential(*layers)

# TODO: generalize the number of layers
class MBEFeedForward(MBELogisticRegression):

    def __init__(self, input_size, pos_weight=None, n_units=128, num_layers = 1):
        super().__init__(input_size, pos_weight)
        self.n_units = n_units
        self.pos_weight = pos_weight
        self.linear = linear_layers(input_size, n_units, num_layers)

class LSTM(nn.Module):
    def __init__(self, input_size, n_hidden_layers, n_stacked_lstms, pos_weight):
        super().__init__()
        self.n_hidden_layers = n_hidden_layers
        self.n_stacked_lstms = n_stacked_lstms
        self.pos_weight = pos_weight
        self.lstm = nn.LSTM(input_size, n_hidden_layers, n_stacked_lstms, batch_first=True)
        self.fc = nn.Linear(input_size, 1, dtype = torch.float64)

    def forward(self, x):
        """
        This will always return the probabilites
        """
        x, _ = self.lstm(x)
        return torch.sigmoid(self.fc(x))
    
    def predict(self,x):
        p = self.forward(x)
        return p/(1-p) / self.pos_weight

class LitMBE(L.LightningModule):

    def __init__(self, model, lr=1e-5, pos_weight = 1):
        super().__init__()
        self.model = model
        self.loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.spearman = stats.spearmanr 
        self.lr = lr
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.type(torch.float64)
        y = y.type(torch.float64)
        logits = self.model(x).squeeze()
        loss = self.loss(logits, y)
        self.log('train_loss', loss, on_step = True, on_epoch = True, prog_bar = True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.type(torch.float64)
        y = y.type(torch.float64)
        with torch.no_grad():
            le = self.model.predict(x).squeeze().cpu().numpy()
            spearman = self.spearman(le, y.cpu().numpy()).statistic
            self.log('val_spearman', spearman, on_step = False, on_epoch = True, prog_bar = True)
        

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)