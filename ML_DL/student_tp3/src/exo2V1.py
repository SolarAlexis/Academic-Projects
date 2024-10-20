import os

from utils import RNN, RNNV2, device,SampleMetroDataset, collate_fn, save_checkpoint, load_checkpoint
import torch
from torch.utils.data import DataLoader
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

matrix_train, matrix_test = torch.load(open(PATH+"hzdataset.pch","rb"), weights_only=True)
ds_train = SampleMetroDataset(matrix_train[:, :, :CLASSES, :DIM_INPUT], length=LENGTH)
ds_test = SampleMetroDataset(matrix_test[:, :, :CLASSES, :DIM_INPUT], length = LENGTH, stations_max = ds_train.stations_max)
data_train = DataLoader(ds_train,batch_size=BATCH_SIZE,shuffle=True, collate_fn=collate_fn)
data_test = DataLoader(ds_test, batch_size=BATCH_SIZE,shuffle=False, collate_fn=collate_fn)

#  TODO:  Question 2 : prédiction de la ville correspondant à une séquence

device = device
print(f"Device: {device}")

model = RNNV2(LENGTH, BATCH_SIZE, DIM_INPUT, CLASSES, latent_size).to(device)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

num_epochs = 50

# Définit le SummaryWriter
script_dir = os.getcwd()
log_dir = os.path.join(script_dir, "exo2_V2")
os.makedirs(log_dir, exist_ok=True)
logger = SummaryWriter(os.path.join(log_dir))

model_dir = os.path.join(script_dir, "model_exo2_V2")
os.makedirs(model_dir, exist_ok=True)
start_epoch = load_checkpoint(model, optimizer, model_dir)

for epoch in tqdm(range(start_epoch, start_epoch + num_epochs)):
    
    cumloss = 0
    correct = 0
    total = 0
    model.train()
    for Xbatch, Ybatch in data_train:
        latents = model(Xbatch)
        Y_pred = model.decoder(latents[-1])
        correct += (torch.argmax(Y_pred, dim=1) == Ybatch).sum().item()
        total += Xbatch.shape[0]
        loss = loss_fn(Y_pred, Ybatch)
        cumloss += loss.item()
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
           
    avg_loss = cumloss / total  # 299 training batches
    logger.add_scalar("train_loss", avg_loss, epoch)
    accuracy = correct / total # 299 training batches
    logger.add_scalar("train_accuracy", accuracy, epoch)
    
    cumloss = 0
    correct = 0
    total = 0
    model.eval()
    for Xbatch, Ybatch in data_test:
        latents = model(Xbatch)
        Y_pred = model.decoder(latents[-1])
        correct += (torch.argmax(Y_pred, dim=1) == Ybatch).sum().item()
        total += Xbatch.shape[0]
        loss = loss_fn(Y_pred, Ybatch)
        cumloss += loss.item()
    
    avg_loss = cumloss / total # 116 test batches
    logger.add_scalar("test_loss", avg_loss, epoch)
    accuracy = correct / total
    logger.add_scalar("test_accuracy", accuracy, epoch)

save_checkpoint(model, optimizer, epoch, model_dir)