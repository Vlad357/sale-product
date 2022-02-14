import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from sklearn.model_selection import train_test_split
torch.autograd.requires_grad=True
from tqdm.notebook import tqdm
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
SEED = 29
torch.manual_seed(SEED)

data = pd.read_csv('data_for_forecast.csv').drop(columns = ['Unnamed: 0', 'date'])

x = np.array(data.drop(columns = 'sales'), dtype = np.float32)
y = np.array(data['sales'], dtype = np.float32)
scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)
train_features, test_features, train_target, test_target = train_test_split(torch.from_numpy(x).view(len(x), 1, 20), 
                                                                            torch.from_numpy(y), 
                                                                            shuffle = False)

train_dataset = TensorDataset(train_features, train_target)
test_dataset = TensorDataset(test_features, test_target)

train_loader = DataLoader(train_dataset, batch_size = 3000, shuffle = False)
test_loader = DataLoader(test_dataset, batch_size = 3000, shuffle = False)

class Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout
        
        self.norm = nn.BatchNorm1d(num_features = input_dim)
        self.norm1 = nn.BatchNorm1d(num_features = hidden_dim)
        
        self.cnn1 = nn.Conv1d(1, input_dim, 1)
        self.cnn2 = nn.Conv1d(input_dim, hidden_dim, 5)
        self.cnn3 = nn.Conv1d(hidden_dim, hidden_dim, 5)
        self.cnn4 = nn.Conv1d(hidden_dim, hidden_dim, 5)
        self.cf1 = nn.Linear(5 * 8, output_dim)
        
    def forward(self, x):
        out = self.norm(self.cnn1(x))
        out = self.norm1(self.cnn2(out))
        out = self.norm1(self.cnn3(out))
        out = self.norm1(self.cnn4(out))
        out = out.view(out.size(0), -1)
        out = self.cf1(out)
        
        return(out)

INPUT_DIM = 20
HIDDEN_DIM = 5
OUTPUT_DIM = 1
DROPOUT = 0.2
model = Model(INPUT_DIM, 
              HIDDEN_DIM, 
              OUTPUT_DIM, 
              DROPOUT)
EPOCHS = 1
optim = torch.optim.Adam(model.parameters(), lr = 0.01)
criterion = nn.MSELoss()

def train(model, loader):
    loss_epoch = 0
    iteration = 0
    for train, target in tqdm(loader):
        iteration +=1
        optim.zero_grad()
        out = model(train)
        loss = criterion(out, target)
        loss.backward()
        optim.step()
        loss_epoch += loss.item()
    return loss_epoch/iteration

def test_val(model, loader):
    loss_epoch = 0
    iteration = 0
    for test, target in tqdm(loader):
        iteration +=1
        out = model(test)
        loss = criterion(out, target)
        loss_epoch += loss.item()
    return loss_epoch/iteration

loss_t = 0
loss_list = []
val_loss_list = []
for epoch in range(EPOCHS):
    loss = train(model, train_loader)
    val_loss = test_val(model, test_loader)
    loss_list.append(loss)
    val_loss_list.append(val_loss)
    print(f'loss: {loss} | validation loss: {val_loss}')

torch.save(model ,'models/torch_cnn')

model = torch.load('models/torch_cnn')

pred = model(test_features).detach().numpy()
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
pred_r = scaler.fit_transform(np.array(pred))
scaler2 = MinMaxScaler()
fet = scaler2.fit_transform(np.array(test_target[:300]).reshape(300,-1))

shift = 0 
for i in np.arange(0, 30, 0.01):
    array = (pred_r[:300] * 15) - i
    
    if min(array) > min(fet) and max(array) < max(fet):
        shift = i

plt.plot(fet, color = 'red', linewidth=1.7, label = 'true sale')
plt.plot((pred_r[:300] * 15) - shift, label = 'forecast sale')
plt.legend()
plt.show()