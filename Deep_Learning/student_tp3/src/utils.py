import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RNN(nn.Module):
    def __init__(self, length, batch_size, input_size, output_size, latent_size):
        super().__init__()
        # dimension initialisation
        self.length = length
        self.batch_size = batch_size
        self.input_size = input_size
        self.output_size = output_size
        self.latent_size = latent_size
        
        # parameters initialisation
        self.Wi = nn.Parameter(torch.rand(self.latent_size, self.input_size) * 2 - 1)
        self.Wh = nn.Parameter(torch.rand(self.latent_size, self.latent_size) * 2 - 1)
        self.bh = nn.Parameter(torch.rand(self.latent_size) * 2 - 1)
        self.Wd = nn.Parameter(torch.rand(self.output_size, self.latent_size) * 2 - 1)
        self.bd = nn.Parameter(torch.rand(self.output_size) * 2 - 1)
        
    def one_step(self, x, h):
        return torch.tanh(x @ self.Wi.T + h @ self.Wh.T + self.bh)
    
    def forward(self, X, h0=None):
        
        batch_size = X.shape[0]
        
        if h0 == None:
            h0 = torch.zeros(batch_size, self.latent_size, requires_grad=True, device=X.device)
        
        latent_spaces = [h0]
        for i in range(X.shape[1]):
            latent_spaces.append(self.one_step(X[:, i, :], latent_spaces[-1]))
        
        return torch.cat([h.unsqueeze(0) for h in latent_spaces[1:]], dim=0)
    
    def decoder(self, h):
        return torch.sigmoid(h @ self.Wd.T + self.bd)
    
class RNNV2(nn.Module):
    def __init__(self, length, batch_size, input_size, output_size, latent_size):
        super().__init__()
        # dimension initialisation
        self.length = length
        self.batch_size = batch_size
        self.input_size = input_size
        self.output_size = output_size
        self.latent_size = latent_size
        
        self.linear1 = nn.Linear(self.input_size, self.latent_size)
        self.linear2 = nn.Linear(self.latent_size, self.latent_size)
        self.linear3 = nn.Linear(self.latent_size, self.output_size)
        
        
    def one_step(self, x, h):
        return torch.tanh(self.linear1(x) + self.linear2(h))
    
    def forward(self, X, h0=None):
        
        batch_size = X.shape[0]
        
        if h0 == None:
            h0 = torch.zeros(batch_size, self.latent_size, requires_grad=True, device=X.device)
        
        latent_spaces = [h0]
        for i in range(X.shape[1]):
            latent_spaces.append(self.one_step(X[:, i, :], latent_spaces[-1]))
        
        return torch.cat([h.unsqueeze(0) for h in latent_spaces[1:]], dim=0)
    
    def decoder(self, h):
        return torch.sigmoid(self.linear3(h))
    
    def forecast_on_step(self, X):
        output = torch.zeros(X.shape[0], X.shape[2], X.shape[3])
        for i in range(X.shape[2]):
            x = X[:, :, i, :]
            
            latent = self.forward(x)
            y = self.decoder(latent[-1])
            output[:, i, :] = y
        return output
    
    def forecast(self, X):
        X_temp = X
        X_temp = X_temp.to(device)
        for _ in range(self.length):
            y_pred = self.forecast_on_step(X_temp)
            y_pred = y_pred.to(device)
            X_temp = X_temp[:, 1:, :, :]
            X_temp = torch.cat((X_temp, y_pred.unsqueeze(1)), dim=1)
        
        return X_temp

class SampleMetroDataset(Dataset):
    def __init__(self, data,length=20,stations_max=None):
        """
            * data : tenseur des données au format  Nb_days x Nb_slots x Nb_Stations x {In,Out}
            * length : longueur des séquences d'exemple
            * stations_max : normalisation à appliquer
        """
        self.data, self.length= data, length
        ## Si pas de normalisation passée en entrée, calcul du max du flux entrant/sortant
        self.stations_max = stations_max if stations_max is not None else torch.max(self.data.view(-1,self.data.size(2),self.data.size(3)),0)[0]
        ## Normalisation des données
        self.data = self.data / self.stations_max
        self.nb_days, self.nb_timeslots, self.classes = self.data.size(0), self.data.size(1), self.data.size(2)

    def __len__(self):
        ## longueur en fonction de la longueur considérée des séquences
        return self.classes*self.nb_days*(self.nb_timeslots - self.length)

    def __getitem__(self,i):
        ## transformation de l'index 1d vers une indexation 3d
        ## renvoie une séquence de longueur length et l'id de la station.
        station = i // ((self.nb_timeslots-self.length) * self.nb_days)
        i = i % ((self.nb_timeslots-self.length) * self.nb_days)
        timeslot = i // self.nb_days
        day = i % self.nb_days
        return self.data[day,timeslot:(timeslot+self.length),station],station

class ForecastMetroDataset(Dataset):
    def __init__(self, data,length=20,stations_max=None):
        """
            * data : tenseur des données au format  Nb_days x Nb_slots x Nb_Stations x {In,Out}
            * length : longueur des séquences d'exemple
            * stations_max : normalisation à appliquer
        """
        self.data, self.length= data,length
        ## Si pas de normalisation passée en entrée, calcul du max du flux entrant/sortant
        self.stations_max = stations_max if stations_max is not None else torch.max(self.data.view(-1,self.data.size(2),self.data.size(3)),0)[0]
        ## Normalisation des données
        self.data = self.data / self.stations_max
        self.nb_days, self.nb_timeslots, self.classes = self.data.size(0), self.data.size(1), self.data.size(2)

    def __len__(self):
        ## longueur en fonction de la longueur considérée des séquences
        return self.nb_days*(self.nb_timeslots - self.length)

    def __getitem__(self,i):
        ## Transformation de l'indexation 1d vers indexation 2d
        ## renvoie x[d,t:t+length-1,:,:], x[d,t+1:t+length,:,:]
        timeslot = i // self.nb_days
        day = i % self.nb_days
        return self.data[day,timeslot:(timeslot+self.length-1)],self.data[day,(timeslot+1):(timeslot+self.length)]

def collate_fn(batch):
    Xbatch, Ybatch = zip(*batch)  # Sépare les features (Xbatch) et labels (Ybatch)
    Xbatch = torch.stack(Xbatch).to(device)  # Transfère Xbatch sur le GPU
    Ybatch = torch.tensor(Ybatch).to(device)  # Transfère Ybatch sur le GPU
    return Xbatch, Ybatch

def collate_fn2(batch):
    Xbatch, Ybatch = zip(*batch)  # Sépare les features (Xbatch) et labels (Ybatch)
    Xbatch = torch.stack(Xbatch).to(device)  # Transfère Xbatch sur le GPU
    Ybatch = torch.stack(Ybatch).to(device)  # Transfère Ybatch sur le GPU
    return Xbatch, Ybatch

def save_checkpoint(model, optimizer, epoch, model_dir):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch + 1
    }
    save_path = os.path.join(model_dir, f"checkpoint_epoch_{epoch+1}.pth")
    torch.save(checkpoint, save_path)
    print(f"Model saved at {save_path}")
    
def load_checkpoint(model, optimizer, model_dir):
    # Trouver le dernier checkpoint enregistré
    checkpoints = [f for f in os.listdir(model_dir) if f.startswith("checkpoint_epoch_")]
    if checkpoints:
        checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))  # Trier par numéro d'epoch
        last_checkpoint = checkpoints[-1]  # Prendre le plus récent
        checkpoint = torch.load(os.path.join(model_dir, last_checkpoint), weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        print(f"Loaded model from {last_checkpoint}, starting from epoch {epoch}")
        return epoch
    else:
        print("No checkpoint found, starting from epoch 0")
        return 0