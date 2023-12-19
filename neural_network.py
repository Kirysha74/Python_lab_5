import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date
from tqdm import tqdm

# Sequence length of input
WINDOW_SIZE = 14
# Number of hidden layer neurons in LSTM
HIDDEN_SIZE = 128
# Number of LSTMs
NUM_LSTMS = 4

BATCH_SIZE = 32
EPOCHS = 100

df = pd.read_csv(r"C:\Users\user\Desktop\Python\Practice\dataset.csv", delimiter = ",", names = ["Date", "Rate"])

train_df = df[(df['Date'] >= '1998-01-01') & 
              (df['Date'] < '2021-01-01')]
test_df = df[(df['Date'] >= '2021-01-01') & 
             (df['Date'] < '2022-01-01')]

train_series = train_df['Rate'].to_list()[::-1]
train_time = train_df['Date'].to_list()[::-1]
train_time = [date.fromisoformat(time) for time in train_time]
test_series = test_df['Rate'].to_list()[::-1]
test_time = test_df['Date'].to_list()[::-1]
test_time = [date.fromisoformat(time) for time in test_time]

plt.figure(figsize=(10, 4))
plt.plot(train_time, train_series)
plt.xlabel('Days', fontsize=12)
plt.ylabel('Price in USD', fontsize=12)
plt.title(f'Stock', fontsize=18)
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(test_time, test_series)
plt.xlabel('Days', fontsize=12)
plt.ylabel('Price in USD', fontsize=12)
plt.title(f'Stock', fontsize=18)
plt.show()

class SeriesDataset(Dataset):
  def __init__(self, series, window_size):
    self.series = series
    self.window_size = window_size

  def __len__(self):
    return len(self.series) - self.window_size

  def __getitem__(self, idx):
    sequence = self.series[idx: idx + self.window_size]
    label = self.series[idx + self.window_size]

    return torch.tensor(sequence).unsqueeze(-1), torch.tensor(label).unsqueeze(-1)

train_dataset = SeriesDataset(train_series, WINDOW_SIZE)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataset = SeriesDataset(test_series, WINDOW_SIZE)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

class LSTM_Model(nn.Module):
  def __init__(self, hidden_size, num_layers):
    super(LSTM_Model, self).__init__()

    self.relu = nn.ReLU()

    self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size,
                        num_layers=num_layers, batch_first=True)
    
    self.fc1 = nn.Linear(hidden_size, 128)
    self.fc2 = nn.Linear(128, 1)


  def forward(self, x):
    x, _ = self.lstm(x)
    x = x[:, -1]
    x = self.fc1(x)
    x = self.relu(x)
    x = self.fc2(x)
    return x

model = LSTM_Model(HIDDEN_SIZE, NUM_LSTMS)


optimizer = optim.Adam(model.parameters())
loss_function = nn.L1Loss()

for epoch in range(EPOCHS):
  loop = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{EPOCHS}', colour='green')
  for sequence, label in loop:
    
    output = model(sequence)
    loss = loss_function(label, output)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loop.set_postfix({'L1 Loss': loss.item()})

predictions = []
model.eval()

loop = tqdm(test_dataloader, colour='green')
for sequence, label in loop:
  
  output = model(sequence)

  for val in output:
    predictions.append(val.item())

plt.figure(figsize=(10, 4))
plt.plot(test_time[WINDOW_SIZE:], test_series[WINDOW_SIZE:], label='Original')
plt.plot(test_time[WINDOW_SIZE:], predictions, label='Prediction')
plt.legend()
plt.xlabel('Days', fontsize=12)
plt.ylabel('Price in USD', fontsize=12)
plt.title(f'Stock', fontsize=18)
plt.show()