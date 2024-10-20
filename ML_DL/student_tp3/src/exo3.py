import os

from utils import RNN, RNNV2, device,ForecastMetroDataset, collate_fn2, save_checkpoint, load_checkpoint

from torch.utils.data import  DataLoader
import torch

from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

# Nombre de stations utilisé
CLASSES = 10
#Longueur des séquences
LENGTH = 20
# Dimension de l'entrée (1 (in) ou 2 (in/out))
DIM_INPUT = 2
#Taille du batch
BATCH_SIZE = 32
# dimension de l'espace latent
latent_size = 1024

PATH = "D:\\amal\\dataset\\"


matrix_train, matrix_test = torch.load(open(PATH+"hzdataset.pch", "rb"), weights_only=True)
ds_train = ForecastMetroDataset(
    matrix_train[:, :, :CLASSES, :DIM_INPUT], length=LENGTH)
ds_test = ForecastMetroDataset(
    matrix_test[:, :, :CLASSES, :DIM_INPUT], length=LENGTH, stations_max=ds_train.stations_max)
data_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn2)
data_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn2)

#  TODO:  Question 3 : Prédiction de séries temporelles

model = RNNV2(LENGTH-1, BATCH_SIZE, DIM_INPUT, DIM_INPUT, latent_size).to(device)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

num_epochs = 10

# Définit le SummaryWriter
script_dir = os.getcwd()
log_dir = os.path.join(script_dir, "exo3_V1")
os.makedirs(log_dir, exist_ok=True)
logger = SummaryWriter(os.path.join(log_dir))

model_dir = os.path.join(script_dir, "model_exo3_V1")
os.makedirs(model_dir, exist_ok=True)
start_epoch = load_checkpoint(model, optimizer, model_dir)

for epoch in tqdm(range(start_epoch, start_epoch + num_epochs)):
    
    total = 0
    cumloss = 0
    model.train()
    for Xbatch, Ybatch in data_train:
        Y_pred = model.forecast(Xbatch)
        loss = loss_fn(Y_pred, Ybatch)
        cumloss += loss.item()
        total += Xbatch.shape[0] * (LENGTH-1)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
           
    avg_loss = cumloss / total
    logger.add_scalar("train_loss", avg_loss, epoch)
    
    cumloss = 0
    total = 0
    model.eval()
    for Xbatch, Ybatch in data_test:
        Y_pred = model.forecast(Xbatch)
        loss = loss_fn(Y_pred, Ybatch)
        cumloss += loss.item()
        total += Xbatch.shape[0] * (LENGTH-1)
    
    avg_loss = cumloss / total
    logger.add_scalar("test_loss", avg_loss, epoch)

save_checkpoint(model, optimizer, epoch, model_dir)