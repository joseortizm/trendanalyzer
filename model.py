import torch
import numpy as np
import pandas as pd
from tqdm import tqdm 
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import torch.nn as nn

if torch.cuda.is_available():
    device = torch.device('cuda')
    print("using...", device)
elif torch.backends.mps.is_available():
    device = torch.device('mps') #my macbook pro :) 
    print("using...", device)
else:
    device = torch.device("cpu")
    print("using...", device)

#main objective: create your own class to manage DataFrame
class getDataset(Dataset):
    def __init__(self, root_dir, csv_file, delimiter=','):
        df = pd.read_csv(root_dir+csv_file, delimiter=delimiter)
        self.data = df 
        
        y = df[df.columns[-1]] 
        x = df.drop(columns=["RowNumber", "CustomerId", "Surname", "Exited"]) 
        
        self.x = x 
        self.y = y 
        self.root_dir = root_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        idx = idx - 1
        feature = self.data.iloc[idx, :-1]
        label= self.data.iloc[idx, -1]
        return feature, label
    
    def getall(self):
        return self.data 
    
    def get_x(self):
        return self.x 

    def get_y(self):
        return self.y

    def getTransform(self):
        X = self.x 
        y = self.y  

        X = pd.get_dummies(X)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
        print("X_train.shape[1]",  X_train.shape[1])
        
        tensors_X_train = torch.from_numpy(X_train).float().to(device)
        tensors_y_train = torch.from_numpy(y_train.values).float().to(device)
        tensors_y_train =  tensors_y_train[:, None]

        tensors_X_test = torch.from_numpy(X_test).float().to(device)
        tensors_y_test = torch.from_numpy(y_test.values).float().to(device)
        tensors_y_test = tensors_y_test[:, None]

        return  tensors_X_train, tensors_y_train, tensors_X_test, tensors_y_test 

#usage examples
DATASET_PATH = "../datasets/Bank_Churn/"
dataset = getDataset(DATASET_PATH, "Churn_Modelling.csv")

#a, b = dataset[3]
#print(a)
#print("b", b)

#print(dataset.getall())

#print("data.get_x:", dataset.get_x())
#print("data.get_y:", dataset.get_y())

tensors_X_train, tensors_y_train, tensors_X_test, tensors_y_test = dataset.getTransform()
print("tensors_X_train.shape:", tensors_X_train.shape)
print("tensors_X_train.shape[1]:", tensors_X_train.shape[1])
print("dataset.get_x().shape:", dataset.get_x().shape)

print("----")
print(tensors_X_train.shape)
print(tensors_y_train.shape)
print(tensors_X_test.shape)
print(tensors_y_test.shape)

class Net(nn.Module):
    def __init__(self, n_input):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(n_input,15)
        self.linear2 = nn.Linear(15, 8)
        self.linear3 = nn.Linear(8, 1)


    def forward(self, x):
        x = torch.sigmoid(input = self.linear1(x))
        x = torch.sigmoid(input = self.linear2(x))
        x = torch.sigmoid(input = self.linear3(x))
        return x

learning_rate = 0.001
#n_input = X_train.shape[1]
n_input = tensors_X_train.shape[1]
#print(n_input)
net = Net(n_input = n_input)
net = net.to(device)
print(net)
optimizer = torch.optim.Adam(params=net.parameters(), lr=learning_rate)
loss_fn = nn.BCELoss()
epochs = 100000
correct_predictions = 0

for epoch in tqdm(range(1, epochs+1)):
    y_pred = net(tensors_X_train)
    loss = loss_fn(y_pred, tensors_y_train)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if epoch % 10000 == 0:
        print("Epoch:", epoch, "| Loss:", round(loss.item(), 4))


    with torch.no_grad():
        y_pred = net(tensors_X_test)
        y_pred_class = y_pred.round()
        correct_predictions = (y_pred_class == tensors_y_test).sum()
        accuracy = (correct_predictions*100)/float(len(tensors_y_test))
        if epoch % 10000 == 0:
            print("Acc:", accuracy.item())

print("Accuracy:",round(accuracy.item(), 2))





