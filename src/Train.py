import librosa
import numpy as np
import pandas as pd
import math, random
from tqdm import tqdm

# Read audio file
import torchaudio
from torchaudio import transforms
from IPython.display import Audio

# Dataloader
from torch.utils.data import DataLoader, Dataset, random_split

# CNN
import torch
from torchsummary import summary
import torch.nn as nn
import torchvision.models as models

# Custom functions



df_train = pd.read_csv('train_dataset.csv')
df_test =  pd.read_csv('test_dataset.csv')


