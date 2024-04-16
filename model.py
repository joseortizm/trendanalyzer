import torch
import numpy as np
import pandas as pd
from tqdm import tqdm 
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler


if torch.cuda.is_available():
    device = torch.device('cuda')
    print("using...", device)
elif torch.backends.mps.is_available():
    device = torch.device('mps')
    print("using...", device)
else:
    device = torch.device("cpu")
    print("using...", device)


class getDataset(Dataset):
    def __init__(self, root_dir, csv_file, delimiter=';', transform=None):
        #df = pd.read_csv(root_dir+csv_file, delimiter=delimiter, header=None)
        df = pd.read_csv(root_dir+csv_file, delimiter=delimiter)
        #df.drop(df.index[0], inplace=True)  
        df.drop(columns=df.columns[0], inplace=True)  
        
        self.data = df 
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        #row = self.data.iloc[idx]        
        row = np.array([idx])
        return row
        #feature = self.data.iloc[idx, :-1]
        #label= self.data.iloc[idx, -1]
        #return feature, label

    
    def getall(self):
        return self.data 
    
    def get_x(self):
        x = self.data.drop(self.data.columns[-1], axis=1)  
        return x 

    def get_y(self):
        y = self.data[self.data.columns[-1]]
        return y

    def getTensor(self, dataf):
        d = pd.get_dummies(dataf)
        escalador = StandardScaler()
        d = escalador.fit_transform(d)
        dataTensor = torch.from_numpy(d).float().to(device)
        return dataTensor




DATASET_PATH = "../datasets/birthCertificates/"


dataset = getDataset(DATASET_PATH, "29_OTI_2024_ENE.csv")
print("getall:")
print(dataset.getall())
print("X:")
print(dataset.get_x())
print("Y:")
print(dataset.get_y())
######
datos_x = dataset.get_x()
tensores_x = dataset.getTensor(datos_x)
print(tensores_x)
print(tensores_x.shape)
