import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import lightning as L
import os
from scipy import stats

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

def linear_layers(in_features, out_fatures, num_layers = 1, act = True):

    layers = []
    
    # Add the input layer and first ReLU activation
    layers.append(nn.Linear(in_features, out_fatures, dtype=torch.float32))
    
    decrement = int((in_features - out_fatures)/num_layers)
    in_t = in_features
    out_t = out_fatures

    # Add the hidden layers
    for x in range(1, num_layers):
        if act: 
            layers.append(nn.ReLU())

        if x == num_layers - 1 and out_t != out_fatures:
            out_t = out_fatures

        layers.append(nn.Linear(in_t, out_t, dtype=torch.float32))
        
        in_t = out_t
        out_t = out_t - decrement
    
    return layers

class MBEFeedForward(MBELogisticRegression):

    def __init__(self, input_size, pos_weight=None, n_units=128, num_layers = 1):
        super().__init__(input_size, pos_weight)
        self.n_units = n_units
        self.pos_weight = pos_weight
        self.linear = nn.Sequential(*linear_layers(input_size, n_units, num_layers))

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, n_stacked_lstms,  pos_weight = 1):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_stacked_lstms = n_stacked_lstms
        self.input_size = input_size
        self.pos_weight = pos_weight
        self.lstm = nn.LSTM(input_size, hidden_size, n_stacked_lstms, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1, dtype = torch.float32)

    def forward(self, x):
        """
        This will always return the probabilites
        """
        device = x.device
        h0 = torch.zeros(self.n_stacked_lstms, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.n_stacked_lstms, x.size(0), self.hidden_size).to(device)

        out, _ = self.lstm(x, (h0, c0))
        out = out.to(torch.float32)

        out = self.fc(out[:, -1, : ])
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

def create_conv_layers(num_layers, n_first_out_ch, kernel_size, stride, layer_out_increment, p_ks, p_s):
    """
    Helper function to create N number of Conv2d layers
    Each "layer" consists of (Conv2d + ReLU) + (Conv2d + ReLU) + Pooling

    p_ks = pooling kernel size
    p_s = pooling stride
    """
    layers = []
    in_t = 50
    out_t = n_first_out_ch
    for i in range(num_layers):
        print("first input: (", in_t,", ", out_t, ", kernel_size=", kernel_size, ", stride= ", stride, ")")
        layers.append(nn.Conv2d(in_t, out_t, kernel_size=kernel_size, stride=stride))
        layers.append(nn.ReLU())

        in_t = out_t
        out_t = out_t * layer_out_increment
        print("second input: (", in_t,", ", out_t, ", kernel_size=", kernel_size, ", stride= ", stride, ")")
        layers.append(nn.Conv2d(in_t, out_t, kernel_size=kernel_size, stride=stride))
        layers.append(nn.ReLU())

        in_t = out_t
        out_t = out_t * layer_out_increment
        print("pooling input: (", p_ks, ", ", p_s, ")")
        layers.append(nn.MaxPool2d(p_ks, p_s))

    return layers

class CNN(nn.Module):
    """
    Class for logistic regression. 
    Returns logits by default, or probabilities if probs=True.
    """
    # TODO: ADD ALL OF THESE PARAMETERS TO THE PARAEMTERS USED IN THE BASH FILE
    def __init__(self, number_of_layers, kernel_size = 3, stride = 1, n_first_out_ch = 100, out_layer_inc = 2, p_ks = 2, p_s = 2, pos_weight = 1):
        super().__init__()
        self.n_first_out_ch = n_first_out_ch # this is the nubmer of outpute channels of the first Conv2d layer
        self.number_of_layers = number_of_layers
        self.kernel_size = kernel_size
        self.stride = stride
        self.pos_weight = pos_weight
        # TODO: I belive that the sequential is used correctly, but make sure
        #self.conv_layers = nn.Sequential(*create_conv_layers(number_of_layers, n_first_out_ch, kernel_size, stride, out_layer_inc, p_ks, p_s))
        
        self.last_out = 500000

        self.conv1 = nn.Conv2d(1, self.n_first_out_ch, self.kernel_size)
        self.conv2 = nn.Conv2d(self.n_first_out_ch, self.last_out, self.kernel_size)
        self.pool = nn.MaxPool2d(p_ks, p_s)
        # self.linear_layers = nn.Sequential(*linear_layers(n_first_out_ch *(out_layer_inc*2*number_of_layers), 1, act = False))
        self.fc1 = nn.Linear(self.last_out*self.kernel_size*self.kernel_size,200)
        self.fc2 = nn.Linear(200,100)
        self.fc3 = nn.Linear(100,50)    

    def forward(self, x):
        """
        Return logits or probabilities
        
        """

        # # TODO: not sure what nn.Sequential returns so this is prob wrong
        # print("x.shape: ", x.shape)
        # out = self.conv_layers(x)
        # print("1. out.shape: ", out.shape)
        # out = self.linear_layers(out)
        # print("2. out.shape: ", out.shape)
        # # TODO: try squeezing out before returning it
        # return out

        print("X SIZE: ", x.shape)
        out = x.unsqueeze(0).permute(1,0,2,3)
        out = self.pool( F.relu(self.conv1(out)))
        print("OUT SIZE 1: ", out.shape)
        out = self.pool( F.relu(self.conv2(out)))
        print("OUT SIZE 2: ", out.shape)
        out = out.view(-1, self.last_out*self.kernel_size*self.kernel_size)
        print("OUT SIZE 3: ", out.shape)
        out = F.relu(self.fc1(out))
        print("OUT SIZE 4: ", out.shape)
        out = F.relu(self.fc2(out))
        print("OUT SIZE 5: ", out.shape)
        out = F.relu(self.fc3(out))
        print("OUT SIZE 6: ", out.shape)
        out = out.permute(1,0)
        print("OUT SIZE 7: ", out.shape)
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
        print("Batch x dimensions: ", x.shape)
        with torch.no_grad():
            le = self.model.predict(x).squeeze().cpu().numpy()
            print("le shape: ", le.shape)
            print("y.cpu().numpy(): ", y.cpu().numpy().shape)
            spearman = self.spearman(le, y.cpu().numpy()).statistic
            self.log('val_spearman', spearman, on_step = False, on_epoch = True, prog_bar = True)
        

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)