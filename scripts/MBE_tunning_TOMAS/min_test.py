import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import lightning as L
import os
from scipy import stats
import sys

a = sys.argv[1]
b = sys.argv[2]
c = sys.argv[3]

print(a)
print(b)
print(c)

