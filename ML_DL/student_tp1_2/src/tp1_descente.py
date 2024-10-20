import os
import shutil

import torch
from torch.utils.tensorboard import SummaryWriter
from tp1 import MSE, Linear, Context


# Les données supervisées
X = torch.randn(50, 13, requires_grad=True, dtype=torch.float64)
Y = torch.randn(50, 3, requires_grad=True, dtype=torch.float64)

# Les paramètres du modèle à optimiser
W = torch.randn(13, 3, requires_grad=True, dtype=torch.float64)
b = torch.randn(3, requires_grad=True, dtype=torch.float64)

epsilon = 0.05

 # Récupére l'emplacement du fichier
script_dir = os.getcwd() 
runs_dir = os.path.join(script_dir, "run_descente")

# Vérifie si le dossier "runs" existe
if os.path.exists(runs_dir):
    # Supprime le dossier "runs" et tout son contenu
    shutil.rmtree(runs_dir)
    
# Recrée le dossier "runs"
os.makedirs(runs_dir, exist_ok=True)

# Définit le SummaryWriter
logger = SummaryWriter(runs_dir)


for n_iter in range(100):
    ##  TODO:  Calcul du forward (loss)
    Y_hat = Linear.apply(X, W, b)
    loss = MSE.apply(Y_hat, Y)
    
    # `loss` doit correspondre au coût MSE calculé à cette itération
    # on peut visualiser avec
    # tensorboard --logdir runs/
    logger.add_scalar('Loss/train', loss.item(), n_iter)

    # Sortie directe
    print(f"Itérations {n_iter}: loss {loss}")

    ##  TODO:  Calcul du backward (grad_w, grad_b)
    loss.backward()
    ##  TODO:  Mise à jour des paramètres du modèle
    with torch.no_grad():
        W -= epsilon * W.grad
        b -= epsilon * b.grad
        
    W.grad.zero_()
    b.grad.zero_()

