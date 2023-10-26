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
from Data_loader import SoundDS
from Model_audio import AudioReg
from Utils import AudioUtil





# -----------------------------
# Training Loop
# ----------------------------
def training(model, train_dl, num_epochs):
  # Loss Function, Optimizer and Scheduler
  criterion = nn.MSELoss()
  optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
  scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001,
                                                steps_per_epoch=int(len(train_dl)),
                                                epochs=num_epochs,
                                                anneal_strategy='linear')

  # Repeat for each epoch
  for epoch in range(num_epochs):
    running_loss = 0.0
    correct_prediction = 0
    total_prediction = 0

    with tqdm(train_dl, unit="batch") as tepoch:

      # Repeat for each batch in the training set
      for i, data in enumerate(tepoch):
          #Refresh tqdm
          tepoch.set_description(f"Epoch {epoch}")

          # Get the input features and target labels, and put them on the GPU
          inputs, labels = data[0].to(device), data[1].to(device)
          # Ensure labels are of float data type
          labels = labels.float()
       

          # Normalize the inputs
          inputs_m, inputs_s = inputs.mean(), inputs.std()
          inputs = (inputs - inputs_m) / inputs_s

          # Zero the parameter gradients
          optimizer.zero_grad()

          # forward + backward + optimize
          outputs = model(inputs)
          
          loss = torch.sqrt(criterion(outputs, labels))
          loss.backward()
          optimizer.step()
          scheduler.step()

          # Keep stats for Loss and Accuracy
          running_loss += loss.item()

          # Get the predicted class with the highest score
          _, prediction = torch.max(outputs,1)
          # Count of predictions that matched the target label
          current_acc =  (prediction == labels).sum().item() / prediction.shape[0]
          correct_prediction += (prediction == labels).sum().item()
          total_prediction += prediction.shape[0]

          #if i % 10 == 0:    # print every 10 mini-batches
          #    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))
          tepoch.set_postfix(loss=loss.item() , accuracy=100. * current_acc)
      # Print stats at the end of the epoch
      num_batches = len(train_dl)
      avg_loss = running_loss / num_batches
      acc = correct_prediction/total_prediction

      
    print(f'Epoch: {epoch}, Average Loss: {avg_loss:.2f}, Average accuracy: {acc:.2f}')

  print('Finished Training')
  


if __name__ == '__main__':

    # Call the dataset
    df_train = pd.read_csv('train_dataset.csv')
    df_test =  pd.read_csv('test_dataset.csv')

    # Create the model and put it on the GPU if available
    model = AudioReg()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    # Check that it is on Cuda
    print(f"[INFO:] device is {next(model.parameters()).device}")

    # Call the dataloader
    ds_train = SoundDS(df=df_train, duration=10000)
    # Create training and validation data loaders
    train_dl = torch.utils.data.DataLoader(ds_train, batch_size=16, shuffle=True)

    # Train the model
    num_epochs= 1000   # Just for demo, adjust this higher.
    training(model, train_dl, num_epochs)