import pandas as pd
# import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import lightning as L
import os
from scipy import stats
import math

# TODO: CHECK IF YOU NEED THIS:
import torch.nn.functional as F

datadir = './out/corrected/counts'
procdir = './out/modelling/processed'

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
        return torch.tensor(counts.loc[1] / counts.loc[0], dtype=torch.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return row['encoded'], row['label'].astype('float32')
    
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

    def __init__(self, data_dir, encode, batch_size = 64):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.encode = encode # encode is the name of the encoding/architecture used

    def setup(self, stage=None):
        if ".pkl" in self.encode:
            self.train = pd.read_pickle(os.path.join(self.data_dir, f'train_{self.encode}'))
            self.eval = pd.read_pickle(os.path.join(self.data_dir, f'eval_{self.encode}'))
            self.test = pd.read_pickle(os.path.join(self.data_dir, f'test_{self.encode}'))
        else:
            self.train = pd.read_parquet(os.path.join(self.data_dir, f'train_{self.encode}'))
            self.eval = pd.read_parquet(os.path.join(self.data_dir, f'eval_{self.encode}'))
            self.test = pd.read_parquet(os.path.join(self.data_dir, f'test_{self.encode}'))

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
        self.linear = nn.Linear(input_size, 1, dtype = torch.float32)
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

def linear_layers(in_features, out_features = 1, num_h_layers = 1, layer_size_var = 'None'):
    print("linear layers")
    """
    Add linear layers to the feed forward network
    layer_size_vay = 
    None: all the layers will have the same input and output
    increase: The layers will increase until the middle and then decrese (this might not be useful?)
    decrese: the layers will decrese all the way down
    """
    # TODO: out features should always be 1
    # TODO: in features would be a constant too because it is determined by the data not the user and the data always has the
    # same number of in features
    layers = []
    
    
    
    if layer_size_var == 'None':
        print("equal")
        layers.append(nn.Linear(in_features, in_features, dtype=torch.float32))
        for x in range(num_h_layers):
            layers.append(nn.ReLU())
            if x == num_h_layers - 1:
                layers.append(nn.Linear(in_features, out_features, dtype=torch.float32))
            else:
                layers.append(nn.Linear(in_features, in_features, dtype=torch.float32))

    elif layer_size_var == 'inc':
        print("INCREASE")
        # TODO: add the number of increments as a parameters
        increment = 100

        in_t = in_features
        out_t = in_t + increment

        layers.append(nn.Linear(in_t, out_t, dtype=torch.float32))

        for i in range(num_h_layers):
            in_t = out_t
            out_t = out_t + increment

            layers.append(nn.ReLU())

            if i == num_h_layers - 1:
                layers.append(nn.Linear(in_t, out_features, dtype=torch.float32))
            else:
                layers.append(nn.Linear(in_t, out_t, dtype=torch.float32))

    elif layer_size_var == 'dec':
        """
        This settting is not very useful since decreasing the number of nodes makes the NN lose a lot of information
        """
        print("DECREASING")
        decrement = int((in_features - out_features)/num_h_layers)
        in_t = in_features
        out_t = in_t - decrement
        print("initial: ", out_t, ", ", in_t)

        layers.append(nn.Linear(in_t, out_t, dtype=torch.float32))

        # Add the hidden layers
        for x in range(0, num_h_layers):
            in_t = out_t
            out_t = out_t - decrement
            
            layers.append(nn.ReLU())

            # TODO: there should be a way to avoid the out_t <= 0 condition, check how to do it
            if (x == num_h_layers - 1 and out_t != out_features) or  out_t <= 0:
                out_t = out_features

            layers.append(nn.Linear(in_t, out_t, dtype=torch.float32))
            print("(out, in): ", out_t, ", ", in_t)
    else: # layer_size_var == 'hybrid'
        delta  = 100 # this is the change of neurons between layers
        even = num_h_layers % 2 == 0
        inc = True

        in_t = in_features
        out_t = in_t + delta

        layers.append(nn.Linear(in_t, out_t, dtype=torch.float32))
        layers.append(nn.ReLU())

        for i in range(num_h_layers):

            in_t = out_t
            print("in and out: ", in_t, ", ", out_t)
            if inc:
                out_t = out_t + delta

                # find the switching point between increasing and decresing
                if i == int(num_h_layers / 2) - 1:
                    inc = False
                    if even:
                        out_t = in_t
                
                layers.append(nn.Linear(in_t, out_t, dtype=torch.float32))
    
            else:
                out_t = out_t - delta
                layers.append(nn.Linear(in_t, out_t, dtype=torch.float32))

            layers.append(nn.ReLU())

        in_t = out_t

        layers.append(nn.Linear(in_t, out_features, dtype=torch.float32))
        pass
    
    return layers

class MBEFeedForward(MBELogisticRegression):

    def __init__(self, input_size, pos_weight=None, num_layers = 1, layer_size_var = 'None'):
        super().__init__(input_size, pos_weight)
        self.pos_weight = pos_weight
        self.layer_size_var = layer_size_var
        self.linear = nn.Sequential(*linear_layers(input_size, num_h_layers=num_layers, layer_size_var=self.layer_size_var))

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, n_stacked_lstms,  pos_weight = 1):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_stacked_lstms = n_stacked_lstms
        self.input_size = input_size
        self.pos_weight = pos_weight
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=n_stacked_lstms, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1, dtype = torch.float32)

    def forward(self, x):
        """
        This will always return the probabilites
        """
        # device = x.device
        # h0 = torch.zeros(self.n_stacked_lstms, x.size(0), self.hidden_size).to(device)
        # c0 = torch.zeros(self.n_stacked_lstms, x.size(0), self.hidden_size).to(device)

        # out, _ = self.lstm(x, (h0, c0))
        # print("first; ", type(x))
        out, _ = self.lstm(x)
        # out = out.to(torch.float32)
        # print(type(out))

        # out = self.fc(out[:, -1, : ])
        out = self.fc(out)
        # print("last: ", type(out), " dims: ", out.shape)
        out = torch.mean(out, dim=1)
        return out
    
    def predict(self,x):
        p = self.forward(x)
        return p/(1-p) / self.pos_weight

###############################################

import scipy.optimize as opt

def rep_equation(h_in, p, d, k, s, rep_left):
    h_out = ((h_in + 2*p - d*(k-1) -1)/s ) +1
    if rep_left == 1:
        # return the minimum variables for the equation to be equal to 50
        return ((h_in + 2*p - d*(k-1) -1)/s ) +1 - 50
    else:
        return rep_equation(h_out, p,d,k,s,rep_left-1)

def equation(vars, target):
    p, d, k, s = vars
    return ((1280 + 2 * p - d * (k - 1) - 1) / s) + 1 - target

def find_minimum_values(target):
    def constraint(vars):
        p, d, k, s = vars
        return s - 1  # s must be greater than or equal to 1 to avoid division by zero

    # Initial guess for the variables
    initial_guess = [1, 1, 1, 1]

    # Bounds for the variables to ensure they are non-negative
    bounds = [(0, None), (0, None), (0, None), (1, None)]

    # Minimize the sum of the variables, subject to the equation constraint
    result = opt.minimize(lambda x: sum(x), initial_guess, constraints={'type': 'eq', 'fun': lambda x: equation(x, target)}, bounds=bounds)

    if result.success:
        p_min, d_min, k_min, s_min = result.x
        return p_min, d_min, k_min, s_min
    else:
        raise ValueError("Optimization failed")

# Example usage
# target_value = 50
# p_min, d_min, k_min, s_min = find_minimum_values(target_value)
# print(f"Minimum values: p = {p_min}, d = {d_min}, k = {k_min}, s = {s_min}")


###############################################

def create_conv_layers(num_layers, input_channel, out_channels, kernel_size, stride, padding, dilation):
    """
    Helper function to create N number of Conv2d layers
    Each "layer" consists of (Conv2d + ReLU) + (Conv2d + ReLU) + Pooling

    p_ks = pooling kernel size
    p_s = pooling stride
    """
    
    """self.conv1 = nn.Conv1d(in_channels=self.in_channels, out_channels=self.out_channels[0], kernel_size=self.ks, stride=self.stride)
    self.fc1 = nn.Linear(self.out_channels[1],64)

    self.conv2 = nn.Conv1d(in_channels=self.out_channels[0], out_channels=self.out_channels[1], kernel_size=self.ks, stride=self.stride)
    self.fc2 = nn.Linear(64,1)

    self.pool = nn.MaxPool1d(kernel_size=self.ks, stride=self.stride)
    self.relu = nn.ReLU()"""
    print("BENINING: nL: ", num_layers, " IN: ", input_channel, " OUT", out_channels, " KS: ", kernel_size, " S: ", stride, " P: ", padding, " D: ", dilation)
    layers = []

    for i in range(num_layers):
        if i == 0:
            print(f"{i}:  IN: ", input_channel, " OUT", out_channels[0], " KS: ", kernel_size, " S: ", stride, " P: ", padding, " D: ", dilation)
            layers.append(nn.Conv1d(in_channels=input_channel, out_channels=out_channels[0], kernel_size=kernel_size, stride=stride))
        else:
            layers.append(nn.Conv1d(in_channels=out_channels[i - 1], out_channels=out_channels[i], kernel_size=kernel_size, stride=stride))
            print(f"{i}:  IN: ", out_channels[i - 1], " OUT", out_channels[i], " KS: ", kernel_size, " S: ", stride, " P: ", padding, " D: ", dilation)
        
        layers.append(nn.BatchNorm1d(out_channels[i]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(p=0.3))
        layers.append(nn.MaxPool1d(kernel_size=kernel_size, stride=stride))
        


        
        # layers.append(nn.Linear(out_channels[-1], out_channels[-1] * 2 ** 2))
        # layers.append(nn.ReLU())
        # layers.append(nn.Linear(out_channels[-1] * 2 ** 2, 1))

    return layers

def calc_Lout(L, p, d, ks, s):
    return math.floor(((L + (2 * p) - (d * (ks - 1)) -1 )/s) + 1)

class CNN(nn.Module):
    """
    Class for logistic regression. 
    Returns logits by default, or probabilities if probs=True.
    """
    def __init__(self, n_layers, out_channels, ks , stride, padding, dilation, pos_weigth = 1):
        super().__init__()
        self.n_layers = n_layers
        self.in_channels = 1280
        self.out_channels = out_channels
        self.ks = ks
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.pos_weight = pos_weigth


        # self.conv1 = nn.Conv1d(in_channels=self.in_channels, out_channels=self.out_channels[0], kernel_size=self.ks, stride=self.stride)
        # self.bn1 = nn.BatchNorm1d(self.out_channels[0])
        self.fc1 = nn.Linear(self.out_channels[-1], self.out_channels[-1] * 2**3)

        # self.conv2 = nn.Conv1d(in_channels=self.out_channels[0], out_channels=self.out_channels[1], kernel_size=self.ks, stride=self.stride)
        # self.bn2 = nn.BatchNorm1d(self.out_channels[1])
        self.fc2 = nn.Linear(self.out_channels[-1] * 2**3,1)

        # self.pool = nn.MaxPool1d(kernel_size=self.ks, stride=self.stride)
        self.relu = nn.ReLU()

        self.convs = nn.Sequential(*create_conv_layers(self.n_layers, self.in_channels, self.out_channels, self.ks, self.stride, self.padding, self.dilation))
        

    def forward(self, x):
        """
        Return logits or probabilities
        
        """
        
        out = x.permute(0,2,1)
        
        # out = self.conv1(out)
        # out = self.relu(out)
        # out = self.pool(out)
        # out = self.bn1(out)
        
        # out = self.conv2(out)
        # out = self.relu(out)
        # out = self.pool(out)
        # out = self.bn2(out)

        out = out.float()
        out = self.convs(out)

        out = out.permute(0,2,1)
        # print("out shape: ", out.shape, " l1: 512")
        out = self.fc1(out)
        out = self.relu(out)
        # print("out shape: ", out.shape, "l2 = 128")
        out = self.fc2(out)
        
        out = torch.mean(out, dim=1)
        
 
        return out

    def predict(self, x):
        """
        Return density ratio, an estimate of log enrichment
        """
        
        # get probability for positive class
        p = self.forward(x)

        # density ratio is p/(1-p), adjusted for class imbalance
        return p/(1-p) / self.pos_weight
    

class LitMBE(L.LightningModule):

    def __init__(self, model, pos_weight = None, lr=1e-5, wd = 1e-5):
        super().__init__()
        self.model = model
        self.loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.spearman = stats.spearmanr 
        self.lr = lr
        self.weight_decay = wd
        self.save_hyperparameters()

    def training_step(self, batch):
        x, y = batch
        x = x.type(torch.float32)
        y = y.type(torch.float32)
        logits = self.model(x).squeeze()
        loss = self.loss(logits, y)
        self.log('train_loss', loss, on_step = True, on_epoch = True, prog_bar = True)
        return loss

    def validation_step(self, batch):
        x, y = batch
        x = x.type(torch.float32)
        y = y.type(torch.float32)
        with torch.no_grad():
            le = self.model.predict(x).squeeze().cpu().numpy()
            spearman = self.spearman(le, y.cpu().numpy()).statistic

            self.log('val_spearman', spearman, on_step = False, on_epoch = True, prog_bar = True)
        

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)