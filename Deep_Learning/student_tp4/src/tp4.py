
import os
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from textloader import *
from generate import *
from tqdm import tqdm

#  TODO: 

DATA_PATH = "D:\\amal\\dataset\\"

def maskedCrossEntropy(output: torch.Tensor, target: torch.LongTensor, padcar: int):
    """
    Calcule la cross-entropie en masquant les positions où il y a des caractères de padding.

    :param output: Tenseur de forme [length x batch_size x output_dim], log-probabilités pour chaque caractère.
    :param target: Tenseur de forme [length x batch_size], valeurs cibles (les vrais caractères).
    :param padcar: Index correspondant au token de padding.
    
    :return: La perte moyenne pondérée par le masque, excluant les caractères de padding.
    """
    # Créer une fonction de perte sans réduction
    criterion = CrossEntropyLoss(reduction='none')

    # On doit d'abord réarranger la sortie pour qu'elle ait la forme [length * batch_size, output_dim]
    # et les cibles pour qu'elles aient la forme [length * batch_size].
    output = output.reshape(-1, output.size(-1))  # (length * batch_size, output_dim)
    target = target.reshape(-1)  # (length * batch_size)

    # Calculer la perte pour chaque élément
    loss = criterion(output, target)  # (length * batch_size)

    # Créer un masque binaire où le padding est 0, les autres caractères sont 1
    mask = (target != padcar).float()  # (length * batch_size)

    # Appliquer le masque à la perte
    masked_loss = loss * mask

    # Calculer la perte moyenne en prenant en compte uniquement les éléments non-paddés
    return masked_loss.sum() / mask.sum()

class RNN(nn.Module):
    def __init__(self, input_size, latent_size, output_size):
        super().__init__()
        # Initialisation des dimensions
        self.input_size = input_size
        self.latent_size = latent_size
        self.output_size = output_size
        
        # Couches linéaires
        self.linear1 = nn.Linear(self.input_size, self.latent_size)
        self.linear2 = nn.Linear(self.latent_size, self.latent_size)
        self.linear3 = nn.Linear(self.latent_size, self.output_size)
    
    def one_step(self, x, h):
        """
        Prend en entrée un vecteur x et un état caché h,
        renvoie le nouvel état caché après un pas de temps.
        """
        return torch.tanh(self.linear1(x) + self.linear2(h))
    
    def forward(self, X, h0=None):
        """
        Prend en entrée un batch de séquences (X) et l'état caché initial (h0),
        renvoie les états cachés à chaque pas de temps.
        X : Tenseur de taille (batch_size, seq_len, input_size)
        h0 : Tenseur de taille (batch_size, latent_size)
        """
        batch_size = X.shape[0]
        seq_len = X.shape[1]
        
        # Initialisation de l'état caché h0
        if h0 is None:
            h0 = torch.zeros(batch_size, self.latent_size, device=X.device)
        
        latent_states = []
        h = h0
        
        # Calcul des états cachés pour chaque pas de temps
        for t in range(seq_len):
            h = self.one_step(X[:, t, :], h)
            latent_states.append(h)
        
        # Retourne les états cachés pour chaque pas de temps
        return torch.stack(latent_states, dim=1)  # Shape: (batch_size, seq_len, latent_size)
    
    def decoder(self, h):
        """
        Décodage de l'état caché en logits (pas de softmax ou sigmoid ici).
        h : Tenseur de taille (batch_size, latent_size)
        Retourne les logits de taille (batch_size, output_size)
        """
        return self.linear3(h)


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, vocab_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Embedding layer to transform indices into vectors of size input_size
        self.embedding = nn.Embedding(vocab_size, input_size)
        
        # Matrices pour les différentes portes et le candidat pour l'état cellulaire
        self.Wf = nn.Linear(input_size + hidden_size, hidden_size)  # Porte d'oubli
        self.Wi = nn.Linear(input_size + hidden_size, hidden_size)  # Porte d'entrée
        self.Wo = nn.Linear(input_size + hidden_size, hidden_size)  # Porte de sortie
        self.Wg = nn.Linear(input_size + hidden_size, hidden_size)  # Candidat pour l'état cellulaire
        
        # Décodeur : couche linéaire pour transformer l'état caché en logits
        self.decoder = nn.Linear(hidden_size, output_size)
        
    def one_step(self, x, h_prev, c_prev):
        # Concaténer l'entrée x (embedding) et l'état caché précédent h_prev
        combined = torch.cat([x, h_prev], dim=1)
        
        # Calculer les portes d'oubli, d'entrée, et de sortie
        ft = torch.sigmoid(self.Wf(combined))  # Porte d'oubli
        it = torch.sigmoid(self.Wi(combined))  # Porte d'entrée
        ot = torch.sigmoid(self.Wo(combined))  # Porte de sortie
        
        # Calculer le candidat pour l'état cellulaire
        gt = torch.tanh(self.Wg(combined))  # Candidat pour l'état cellulaire
        
        # Mettre à jour l'état cellulaire
        c_t = ft * c_prev + it * gt
        
        # Calculer l'état caché
        h_t = ot * torch.tanh(c_t)
        
        return h_t, c_t

    def forward(self, X, h0=None, c0=None):
        """
        X : Tenseur de taille (batch_size, seq_len) (indices des caractères ou mots)
        h0 : Tenseur de taille (batch_size, hidden_size) pour l'état caché initial
        c0 : Tenseur de taille (batch_size, hidden_size) pour l'état cellulaire initial
        """
        batch_size = X.shape[0]
        seq_len = X.shape[1]
        
        # Convertir les indices en embeddings
        X = self.embedding(X)  # X est maintenant de taille (batch_size, seq_len, input_size)
        
        # Initialisation des états cachés et cellulaires
        if h0 is None:
            h0 = torch.zeros(batch_size, self.hidden_size, device=X.device)
        if c0 is None:
            c0 = torch.zeros(batch_size, self.hidden_size, device=X.device)
        
        latent_states = []
        h, c = h0, c0
        
        # Calcul des états cachés pour chaque pas de temps
        for t in range(seq_len):
            h, c = self.one_step(X[:, t, :], h, c)
            latent_states.append(h)
        
        # Retourne les états cachés pour chaque pas de temps
        return torch.stack(latent_states, dim=1), (h, c)  # Shape: (batch_size, seq_len, hidden_size)
    
    def decode(self, h):
        """
        Transforme les états cachés en logits.
        
        :param h: Tenseur de taille (batch_size, seq_len, hidden_size)
        :return: Tenseur de taille (batch_size, seq_len, output_size) (les logits)
        """
        return self.decoder(h)

class GRU_(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, vocab_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Embedding layer to transform indices into vectors of size input_size
        self.embedding = nn.Embedding(vocab_size, input_size)
        
        # Matrices pour les différentes portes
        self.Wz = nn.Linear(input_size + hidden_size, hidden_size)  # zt : porte de mise à jour
        self.Wr = nn.Linear(input_size + hidden_size, hidden_size)  # rt : porte de réinitialisation
        self.W = nn.Linear(input_size + hidden_size, hidden_size)   # ht : état caché actuel
        
        # Décodeur : couche linéaire pour transformer l'état caché en logits
        self.decoder = nn.Linear(hidden_size, output_size)
        
    def one_step(self, x, h_prev):
        
        # S'assurer que x et h_prev sont sur le même device
        assert x.device == h_prev.device, "x et h_prev ne sont pas sur le même device"
        
        """
        Prend un seul pas de temps (x) et l'état caché précédent (h_prev).
        """
        # Concaténer l'entrée x (embedding) et l'état caché précédent h_prev
        combined = torch.cat([x, h_prev], dim=1)

        # Calculer les portes zt et rt
        zt = torch.sigmoid(self.Wz(combined))  # Porte de mise à jour
        rt = torch.sigmoid(self.Wr(combined))  # Porte de réinitialisation

        # Calculer l'état caché candidat h̃t
        combined_reset = torch.cat([x, rt * h_prev], dim=1)
        h_tilde = torch.tanh(self.W(combined_reset))  # Candidat pour l'état caché

        # Calculer l'état caché actuel ht
        ht = (1 - zt) * h_prev + zt * h_tilde

        return ht

    def forward(self, X, h0=None):
        """
        Prend en entrée un batch de séquences (X) et l'état caché initial (h0),
        renvoie les états cachés à chaque pas de temps.
        X : Tenseur de taille (batch_size, seq_len) (indices des caractères ou mots)
        h0 : Tenseur de taille (batch_size, hidden_size)
        """
        batch_size = X.shape[0]
        seq_len = X.shape[1]

        # Convertir les indices en embeddings
        X = self.embedding(X)  # X est maintenant de taille (batch_size, seq_len, input_size)
        
        # Initialisation de l'état caché h0
        if h0 is None:
            h0 = torch.zeros(batch_size, self.hidden_size, device=X.device)
        
        latent_states = []
        h = h0
        
        # Calcul des états cachés pour chaque pas de temps
        for t in range(seq_len):
            h = self.one_step(X[:, t, :], h)  # Passage d'un pas de temps à l'autre
            latent_states.append(h)
        
        # Retourne les états cachés pour chaque pas de temps
        return torch.stack(latent_states, dim=1)  # Shape: (batch_size, seq_len, hidden_size)

    def decode(self, h):
        """
        Transforme les états cachés en logits.
        
        :param h: Tenseur de taille (batch_size, hidden_size)
        :return: Tenseur de taille (batch_size, output_size) (les logits)
        """
        return self.decoder(h)

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

if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    batch_size = 32
    ds = TextDataset(open(DATA_PATH+"trump_full_speech.txt","rb").read().decode(), maxlen=1000)
    data_trump = DataLoader(ds, batch_size= batch_size, shuffle=True, collate_fn=lambda x: pad_collate_fn(x, device=device))
    
    input_size = 50  # Taille de l'embedding
    hidden_size = 128  # Taille de l'état caché
    output_size = len(id2lettre)  # Nombre total de lettres dans le vocabulaire
    learning_rate = 0.001
    num_epochs = 10
    vocab_size = len(lettre2id)
    
    #model_1 = LSTM(input_size, hidden_size, output_size, vocab_size)
    #optimizer_1 = torch.optim.Adam(model_1.parameters(), lr=learning_rate)
    model_2 = GRU_(input_size, hidden_size, output_size, vocab_size).to(device)
    optimizer_2 = torch.optim.Adam(model_2.parameters(), lr=learning_rate)
    loss_fn = maskedCrossEntropy
    
    script_dir = os.getcwd()
    log_dir = os.path.join(script_dir, "log_GRU")
    os.makedirs(log_dir, exist_ok=True)
    logger = SummaryWriter(os.path.join(log_dir))

    model_dir = os.path.join(script_dir, "model_GRU")
    os.makedirs(model_dir, exist_ok=True)
    start_epoch = load_checkpoint(model_2, optimizer_2, model_dir)
    
    for epoch in tqdm(range(start_epoch, start_epoch + num_epochs)):
        model_2.train()  # Mode entraînement
        epoch_loss = 0
        
        for batch in data_trump:
            # Réinitialiser les gradients
            optimizer_2.zero_grad()
            
            # Séparer les données (features) et les cibles
            inputs = batch[:, :-1]  # Tout sauf le dernier caractère
            targets = batch[:, 1:]  # Tout sauf le premier (car prédire le suivant)

            # Passer les données dans le modèle
            h = model_2(inputs)
            outputs = model_2.decode(h)
            
            # Calculer la perte (masked cross entropy)
            loss = loss_fn(outputs, targets, PAD_IX)

            # Backpropagation et mise à jour des poids
            loss.backward()
            optimizer_2.step()

            # Accumuler la perte pour suivre les performances
            epoch_loss += loss.item()
            
        # Ajouter les histogrammes après chaque époque
        for name, param in model_2.named_parameters():
            logger.add_histogram(f"{name}_grad", param.grad, epoch)
            logger.add_histogram(f"{name}_weight", param, epoch)

        # Afficher la perte après chaque époque
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(data_trump)}")
        logger.add_scalar('Loss/train', epoch_loss / len(data_trump), epoch)
        
    save_checkpoint(model_2, optimizer_2, epoch, model_dir)
    