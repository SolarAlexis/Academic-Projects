import logging
import re
from pathlib import Path
from tqdm import tqdm
import numpy as np

from datamaestro import prepare_dataset
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.utils.tensorboard import SummaryWriter

import os

def load_glove_embeddings(filepath, embedding_size):
    word2id = {}
    embeddings = []

    with open(filepath, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype="float32")
            word2id[word] = idx
            embeddings.append(vector)
    
    embeddings = np.array(embeddings)

    # Ajouter un vecteur pour les mots hors vocabulaire (OOV)
    OOVID = len(word2id)
    word2id["__OOV__"] = OOVID
    embeddings = np.vstack((embeddings, np.zeros(embedding_size)))

    return word2id, embeddings

class FolderText(Dataset):
    def __init__(self, classes, folder: Path, tokenizer, load=False):
        self.tokenizer = tokenizer
        self.files = []
        self.filelabels = []
        self.labels = {label: idx for idx, label in enumerate(classes)}

        for label in classes:
            folder_path = folder / label
            if not folder_path.exists():
                continue
            for file in folder_path.glob("*.txt"):
                self.files.append(file.read_text() if load else file)
                self.filelabels.append(self.labels[label])

    def __len__(self):
        return len(self.filelabels)

    def __getitem__(self, ix):
        s = self.files[ix]
        return self.tokenizer(s if isinstance(s, str) else s.read_text(encoding="utf-8")), self.filelabels[ix]
    
def load_imdb_dataset(dataset_path, word2id):
    """
    Charge le dataset IMDB à partir de fichiers locaux.

    Args:
        dataset_path (str): Chemin vers le dossier aclImdb.
        word2id (dict): Dictionnaire de mots vers IDs.

    Returns:
        train_dataset, test_dataset: Datasets PyTorch d'entraînement et de test.
    """
    WORDS = re.compile(r"\S+")
    dataset_path = Path(dataset_path)

    def tokenizer(text):
        return [word2id.get(word, word2id["__OOV__"]) for word in re.findall(WORDS, text.lower())]

    classes = ["positive", "negative"]
    train_dataset = FolderText(classes, dataset_path / "train", tokenizer, load=False)
    test_dataset = FolderText(classes, dataset_path / "test", tokenizer, load=False)

    return train_dataset, test_dataset

def collate_fn(batch):
    """
    Gère le padding et la mise en lot pour un DataLoader.
    Args:
        batch: Liste d'exemples, où chaque exemple est une paire (séquence, label).
    Returns:
        padded_sequences: Tenseur 3D (batch_size, max_seq_len, embedding_size).
        labels: Tenseur 1D (batch_size).
        lengths: Longueurs des séquences avant padding (pour un éventuel usage dans un modèle RNN).
    """
    sequences, labels = zip(*batch)  # Sépare les séquences et les labels
    # Padding des séquences pour qu'elles aient toutes la même longueur
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0.0)
    # Conversion des labels en tenseur
    labels = torch.stack(labels)
    return padded_sequences, labels

def convert_to_embeddings(dataset, embeddings):
    embedded_dataset = []
    for sequence, label in tqdm(dataset, desc="embedding"):  # Parcours des critiques et labels
        # Conversion des IDs en vecteurs d'embeddings
        embedded_sequence = torch.tensor(
            np.array([embeddings[word_id] for word_id in sequence if word_id < len(embeddings)]),
            dtype=torch.float32
        )
        # Conversion du label en tenseur
        label_tensor = torch.tensor(label, dtype=torch.long)  # Labels souvent en long pour la classification
        embedded_dataset.append((embedded_sequence, label_tensor))  # Associer la séquence encodée au label
    return embedded_dataset

class simple_model(nn.Module):
    
    def __init__(self, embedding_size, hidden_size, num_classes):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(embedding_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )
        
    def forward(self, x):
        
        t_hat = torch.mean(x, dim=1)
        y_hat = self.mlp(t_hat)
        
        return y_hat



#-----------------------------------------------------------------------------------------------------------------------------------------------------


glove_path = "D://amal//student_tp8//src//glove.6B.50d.txt"  # Modifiez selon votre emplacement
embedding_size = 50  # Assurez-vous que cela correspond à vos embeddings
word2id, embeddings = load_glove_embeddings(glove_path, embedding_size)

print(f"Nombre de mots : {len(word2id)}")
print(f"Dimension des embeddings : {embeddings.shape}")

dataset_path = "D:/amal/student_tp8/src/aclImdb"
train_dataset, test_dataset = load_imdb_dataset(dataset_path, word2id)

print(f"Taille du dataset d'entraînement : {len(train_dataset)}")
print(f"Taille du dataset de test : {len(test_dataset)}")

embedded_train_dataset = convert_to_embeddings(train_dataset, embeddings)
embedded_test_dataset = convert_to_embeddings(test_dataset, embeddings)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 32
num_epochs = 20
hidden_size = 128
num_classes = 2
model = simple_model(embedding_size, hidden_size, num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
loss_fn = nn.CrossEntropyLoss()
writer = SummaryWriter(log_dir='log_exo1')

train_loader = DataLoader(embedded_train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(embedded_test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

for epoch in tqdm(range(num_epochs), desc="training loop"):
    
    cumloss = 0
    count_batch = 0
    good_pred = 0
    count_element = 0
    for inputs, labels in train_loader:

        inputs, labels = inputs.to(device), labels.to(device)
        
        y_pred = model(inputs)
        loss = loss_fn(y_pred, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        cumloss += loss.item()
        count_batch += 1
        count_element += inputs.shape[0]
        good_pred += torch.sum(torch.argmax(y_pred, dim=1) == labels).item()
        
    train_loss = cumloss / count_batch
    train_accuracy = good_pred / count_element
    writer.add_scalar("train_loss", train_loss, epoch)
    writer.add_scalar("train_accuracy", train_accuracy, epoch)
    
    
    model.eval()
    test_cumloss = 0
    test_good_pred = 0
    test_count_element = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            y_pred = model(inputs)
            loss = loss_fn(y_pred, labels)

            test_cumloss += loss.item()
            test_good_pred += torch.sum(torch.argmax(y_pred, dim=1) == labels).item()
            test_count_element += inputs.shape[0]

    test_loss = test_cumloss / len(test_loader)
    test_accuracy = test_good_pred / test_count_element
    writer.add_scalar("test_loss", test_loss, epoch)
    writer.add_scalar("test_accuracy", test_accuracy, epoch)
        
writer.close()