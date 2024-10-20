import sys
import unicodedata
import string
from typing import List
from torch.utils.data import Dataset, DataLoader
import torch
import re

## Token de padding (BLANK)
PAD_IX = 0
## Token de fin de séquence
EOS_IX = 1

LETTRES = string.ascii_letters + string.punctuation + string.digits + ' '
id2lettre = dict(zip(range(2, len(LETTRES)+2), LETTRES))
id2lettre[PAD_IX] = '<PAD>' ##NULL CHARACTER
id2lettre[EOS_IX] = '<EOS>'
lettre2id = dict(zip(id2lettre.values(),id2lettre.keys()))

def normalize(s):
    """ enlève les accents et les caractères spéciaux"""
    return ''.join(c for c in unicodedata.normalize('NFD', s) if  c in LETTRES)

def string2code(s):
    """prend une séquence de lettres et renvoie la séquence d'entiers correspondantes"""
    return torch.tensor([lettre2id[c] for c in normalize(s)])

def code2string(t):
    """ prend une séquence d'entiers et renvoie la séquence de lettres correspondantes """
    if type(t) !=list:
        t = t.tolist()
    return ''.join(id2lettre[i] for i in t)


class TextDataset(Dataset):
    def __init__(self, text: str, *, maxsent=None, maxlen=None):
        """  Dataset pour les tweets de Trump
            * fname : nom du fichier
            * maxsent : nombre maximum de phrases.
            * maxlen : longueur maximale des phrases.
        """
        maxlen = maxlen or sys.maxsize
        self.phrases = [re.sub(' +',' ',p[:maxlen]).strip() +"." for p in text.split(".") if len(re.sub(' +',' ',p[:maxlen]).strip())>0]
        if maxsent is not None:
            self.phrases=self.phrases[:maxsent]
        self.maxlen = max([len(p) for p in self.phrases])

    def __len__(self):
        return len(self.phrases)

    def __getitem__(self, i):
        return string2code(self.phrases[i])

def pad_collate_fn(samples: List[List[int]], device=torch.device('cpu')):
    #  TODO:  Renvoie un batch à partir d'une liste de listes d'indexes (de phrases) qu'il faut padder.
    # Trouver la longueur maximale des séquences dans ce batch
    max_length = max([len(seq) for seq in samples]) + 1  # +1 pour le token EOS

    # Préparer un batch avec padding
    padded_batch = []
    for seq in samples:
        # Ajouter le token EOS à la fin de chaque séquence
        seq_with_eos = seq.tolist() + [EOS_IX]
        
        # Ajouter du padding pour que chaque séquence ait la même longueur
        padded_seq = seq_with_eos + [PAD_IX] * (max_length - len(seq_with_eos))
        
        # Ajouter la séquence paddée au batch
        padded_batch.append(padded_seq)
    
    # Convertir le batch en tenseur
    batch_tensor = torch.tensor(padded_batch, dtype=torch.long, device=device)
    
    # Transposer pour obtenir [longueur_max, taille_batch]
    return batch_tensor

if __name__ == "__main__":
    test = "C'est. Un. Test."
    ds = TextDataset(test)
    loader = DataLoader(ds, collate_fn=pad_collate_fn, batch_size=3)
    data = next(iter(loader))
    print("Chaîne à code : ", test)
    # Longueur maximum
    assert data.shape == (7, 3)
    print("Shape ok")
    # e dans les deux cas
    assert data[2, 0] == data[1, 2]
    print("encodage OK")
    # Token EOS présent
    assert data[5,2] == EOS_IX
    print("Token EOS ok")
    # BLANK présent
    assert (data[4:,1]==0).sum() == data.shape[0]-4
    print("Token BLANK ok")
    # les chaînes sont identiques
    s_decode = " ".join([code2string(s).replace(id2lettre[PAD_IX],"").replace(id2lettre[EOS_IX],"") for s in data.t()])
    print("Chaîne décodée : ", s_decode)
    assert test == s_decode
    # " ".join([code2string(s).replace(id2lettre[PAD_IX],"").replace(id2lettre[EOS_IX],"") for s in data.t()])
    s_decode = " ".join([code2string(s).replace(id2lettre[PAD_IX],"").replace(id2lettre[EOS_IX],"") for s in data.t()])
    assert test == s_decode
    
    
    from generate import generate
    from tp4 import RNN, GRU_
    
    # Exemple d'utilisation
    input_size = 50   # Taille de l'embedding (input)
    latent_size = 100  # Taille de l'état caché (latent)
    output_size = len(id2lettre)  # Taille de l'espace de sortie (logits)

    # Création du modèle RNN
    rnn_model = RNN(input_size=input_size, latent_size=latent_size, output_size=output_size)
    emb = torch.nn.Embedding(num_embeddings=output_size, embedding_dim=input_size)  # Embedding
    decoder = rnn_model.decoder  # Utilisation de la méthode de décodage du modèle RNN

    eos_token = EOS_IX  # Index du token EOS
    start_sequence = "Hello"

    # Générer une séquence
    generated_text = generate(rnn_model, emb, decoder, eos=eos_token, start=start_sequence, maxlen=100, deterministic=True)
    print("Generated text:", generated_text)
    
    
    
    input_size = 50   # Taille de l'entrée (par exemple, taille de l'embedding)
    hidden_size = 100  # Taille de l'état caché
    output_size = len(id2lettre)  # Taille de l'espace de sortie (nombre total de caractères)

    # Instancier le modèle GRU
    gru_model = GRU_(input_size=input_size, hidden_size=hidden_size)

    # Exemple de batch de séquences (batch_size=32, seq_len=10, input_size=50)
    X = torch.randn(32, 10, input_size)  # 32 séquences de longueur 10 avec des entrées de dimension 50

    # Passage dans le modèle GRU
    output = gru_model(X)

    print(output.shape)  # La sortie sera de taille (batch_size, seq_len, hidden_size)