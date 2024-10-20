from textloader import  string2code, id2lettre
import math
import torch

#  TODO:  Ce fichier contient les différentes fonction de génération

def generate(rnn, decoder, eos, start="", maxlen=200, deterministic=True, device='cpu'):
    """
    Fonction de génération de séquences à partir d'un modèle RNN.
    
    :param rnn: Le modèle GRU utilisé pour générer les séquences.
    :param decoder: La fonction de décodage qui retourne les logits pour chaque sortie possible.
    :param eos: L'index du token EOS.
    :param start: La séquence de départ (chaîne de caractères). Si vide, démarrage à 0.
    :param maxlen: La longueur maximale de la séquence générée.
    :param deterministic: Si True, choisit le caractère le plus probable à chaque étape. 
                          Si False, échantillonne aléatoirement à partir de la distribution de probabilités.
    :param device: Le device sur lequel exécuter la génération ('cpu' ou 'cuda').
    
    :return: La séquence générée sous forme de chaîne de caractères.
    """
    rnn.eval()  # Passer en mode évaluation (désactiver dropout, batchnorm, etc.)
    
    # Initialisation du RNN avec la séquence de départ 'start'
    generated_sequence = []  # Liste pour stocker la séquence générée
    hidden_state = None  # L'état caché initial du RNN (sera passé à chaque étape)

    # Encoder la séquence de départ
    if start:
        input_seq = string2code(start).unsqueeze(0).to(device)  # Convertir start en indices et ajouter une dimension batch
    else:
        input_seq = torch.tensor([[0]], dtype=torch.long).to(device)  # Si 'start' est vide, démarrer avec le token 0

    # Passer la séquence initiale par le RNN pour obtenir l'état caché initial
    output = rnn(input_seq, hidden_state)  # L'input_seq est directement passée à GRU_
    hidden_state = output[:, -1, :]  # Utiliser le dernier état caché

    # Générer à partir de la sortie du RNN, étape par étape
    for _ in range(maxlen):
        # Passer l'état caché à travers le décodeur pour obtenir les logits
        logits = decoder(hidden_state)  # Obtenir les logits pour le dernier élément de la séquence

        if deterministic:
            # Sélection déterministe : choisir le caractère le plus probable
            next_char_index = torch.argmax(logits, dim=-1).item()
        else:
            # Sélection stochastique : échantillonner à partir de la distribution de probabilités
            probabilities = torch.softmax(logits, dim=-1)
            next_char_index = torch.multinomial(probabilities, num_samples=1).item()

        # Ajouter le caractère généré à la séquence
        generated_sequence.append(next_char_index)

        # Arrêter si on a rencontré le token EOS
        if next_char_index == eos:
            break

        # Préparer l'entrée pour l'étape suivante
        next_input = torch.tensor([[next_char_index]], dtype=torch.long).to(device)

        # Passer l'embedding de l'entrée générée par le RNN pour mettre à jour l'état caché
        output = rnn(next_input, hidden_state)
        hidden_state = output[:, -1, :]  # Utiliser le dernier état caché

    # Décoder la séquence d'indices en une chaîne de caractères
    return "".join([id2lettre[idx] for idx in generated_sequence])


def generate_beam(rnn, emb, decoder, eos, k, start="", maxlen=200):
    """
        Génere une séquence en beam-search : à chaque itération, on explore pour chaque candidat les k symboles les plus probables; puis seuls les k meilleurs candidats de l'ensemble des séquences générées sont conservés (au sens de la vraisemblance) pour l'itération suivante.
        * rnn : le réseau
        * emb : la couche d'embedding
        * decoder : le décodeur
        * eos : ID du token end of sequence
        * k : le paramètre du beam-search
        * start : début de la phrase
        * maxlen : longueur maximale
    """
    #  TODO:  Implémentez le beam Search


# p_nucleus
def p_nucleus(decoder, alpha: float):
    """Renvoie une fonction qui calcule la distribution de probabilité sur les sorties

    Args:
        * decoder: renvoie les logits étant donné l'état du RNN
        * alpha (float): masse de probabilité à couvrir
    """
    def compute(h):
        """Calcule la distribution de probabilité sur les sorties

        Args:
           * h (torch.Tensor): L'état à décoder
        """
        #  TODO:  Implémentez le Nucleus sampling ici (pour un état s)
    return compute