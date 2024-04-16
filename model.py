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

    def getTensor(self):
        dato =self.data
        dataTensor = torch.tensor(dato)
        #dataTensor = torch.from_numpy(self.data).float()
        return dataTensor




DATASET_PATH = "../datasets/birthCertificates/"


dataset = getDataset(DATASET_PATH, "29_OTI_2024_ENE.csv")
print(dataset[0])
print("getall:")
print(dataset.getall())
print("X:")
print(dataset.get_x())
print("Y:")
print(dataset.get_y())
#print("tensores:")
#print(torch.tensor(dataset[0], dtype=torch.float).flatten())
#print(torch.tensor(dataset[0], dtype=torch.float))
#data_loader = DataLoader(custom_dataset, batch_size=64, shuffle=True)
#print(dataset[0].to_numpy())
#
#batch_size = 2 
#for batch in dataloader:
#    print(batch)
#print(dataloader)
    #texto_batch, numero1_batch, numero2_batch, texto_batcha, numero1_batchb, numero2_batchc = batch
    ## Imprimimos solo el primer elemento de cada tensor en el batch
    #print("Texto:", texto_batch[0])
    #print("Número 1:", numero1_batch[0])
    #print("Número 2:", numero2_batch[0])
#batch = next(iter(dataloader))
#print(batch)
#a = list(zip(dataset.get_x(), dataset.get_y()))
#print(type(a))
#for i in a:
#    print(i)
#

#train_dataloader = DataLoader(dataset.getall(), batch_size=batch_size, shuffle=True)
#train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
#train_features, train_labels = next(iter(train_dataloader))
#print(f"Feature batch shape: {train_features.size()}")
#print(f"Labels batch shape: {train_labels.size()}")
#print(type(train_dataloader))
#df = dataset.getall()
#train_dataloader = DataLoader(df, batch_size=batch_size, shuffle=True)


#t_x_data = dataset.getTensor()
#print(t_x_data)
##


#for batch in train_dataloader:
#    # batch contendrá un lote de datos
#    # Puedes trabajar con cada lote aquí
#
#    # Por ejemplo, si el lote contiene entradas X y etiquetas y
#    inputs, labels = batch
#
#    # Realiza las operaciones que desees con los datos del lote
#    # Por ejemplo, imprimir la forma de los datos
#    print("Forma de las entradas:", inputs.shape)
#    print("Forma de las etiquetas:", labels.shape)
#for batchIdx, sampledIdx in enumerate(tqdm(train_dataloader, position=0, leave=True)):
#    print("batchIdx:",batchIdx,"sampledIdx:",sampledIdx)
#    #print("sampledIdx.data:", sampledIdx.data)
#    #sampledIdx = sampledIdx.data.numpy()


######


datos_x = dataset.get_x()

datos_x = pd.get_dummies(datos_x)
#print(datos_x.head())
print(datos_x)
escalador = StandardScaler()
datos_x = escalador.fit_transform(datos_x)
print(datos_x)
print(len(datos_x))
datos_y = dataset.get_y() 
print(datos_y)





